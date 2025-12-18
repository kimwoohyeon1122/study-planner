# app.py (통파일) - 현재 사용자 코드 기반 + Render Persistent Disk(DB/업로드/백업 유지) + 백업 30개 유지 자동정리
from collections import defaultdict
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory, abort
)
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, date
import zipfile

app = Flask(__name__)

# =========================
# ✅ Render/장기운영용 설정
# - SECRET_KEY: Render 환경변수 권장
# - DATA_DIR: Render Persistent Disk 마운트 경로(예: /var/data)
# - DB_PATH: DB 파일 위치(기본: DATA_DIR/planner.db)
# - UPLOAD_DIR: 업로드 파일 위치(기본: DATA_DIR/uploads)
# - BACKUP_DIR: 백업 zip 위치(기본: DATA_DIR/backups)
# =========================
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(app.root_path, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DB = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "planner.db"))

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

BACKUP_DIR = os.environ.get("BACKUP_DIR", os.path.join(DATA_DIR, "backups"))
os.makedirs(BACKUP_DIR, exist_ok=True)

# ✅ 백업 보관 개수(기본 30)
try:
    BACKUP_KEEP = int(os.environ.get("BACKUP_KEEP", "30"))
    BACKUP_KEEP = max(1, min(BACKUP_KEEP, 365))
except Exception:
    BACKUP_KEEP = 30

# (선택) 토큰으로 백업 API 호출 허용(로그인 없이 크론에서 호출할 때)
BACKUP_TOKEN = os.environ.get("BACKUP_TOKEN", "")

# 업로드 제한
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


# =========================
# DB helpers
# =========================
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      pw_hash TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS plans (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      subject TEXT NOT NULL,
      pages INTEGER NOT NULL,
      dday TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      plan_id INTEGER NOT NULL,
      log_date TEXT NOT NULL,          -- YYYY-MM-DD
      pages_done INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(plan_id, log_date),
      FOREIGN KEY(plan_id) REFERENCES plans(id)
    )
    """)

    # ✅ 사용자별 이미지 저장(여러 장 누적 저장 가능)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_images (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      image_key TEXT NOT NULL,          -- 예: 'timetable'
      filename TEXT NOT NULL,           -- 저장된 파일명
      uploaded_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # ✅ 기능4: 학습 회고 노트 (날짜별 1개, 사용자별)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reflections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      log_date TEXT NOT NULL,           -- YYYY-MM-DD (회고 날짜)
      goal_met INTEGER NOT NULL DEFAULT 0,   -- 0/1
      obstacles TEXT NOT NULL DEFAULT '',    -- 예: "피로,산만함" 또는 자유서술
      note TEXT NOT NULL DEFAULT '',         -- 한 줄/여러 줄 회고
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(user_id, log_date),
      FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()

init_db()

def current_user_id():
    return session.get("user_id")


def login_required_or_redirect():
    if not current_user_id():
        flash("로그인이 필요합니다.")
        return redirect(url_for("login"))
    return None


# =========================
# Planner 계산용 util
# =========================
def parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def diff_days_inclusive(today: date, dday: date) -> int:
    return (dday - today).days + 1


def calc_target_range(remaining_pages: int, days_left: int):
    if days_left <= 0:
        return None
    raw = remaining_pages / days_left
    mn = int(raw)  # floor
    mx = int(raw) if raw.is_integer() else int(raw) + 1  # ceil
    if mn <= 0 and remaining_pages > 0:
        mn = 1
    return mn, mx


# =========================
# Upload util
# =========================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def ensure_user_upload_dir(user_id: int) -> str:
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_file_size(file_storage) -> int:
    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    return size


# =========================
# ✅ 업로드 파일 서빙 (UPLOAD_DIR이 /static 밖일 때도 동작)
# =========================
@app.route("/uploads/<int:user_id>/<path:filename>")
def uploaded_file(user_id, filename):
    if not current_user_id():
        abort(401)
    if current_user_id() != user_id:
        abort(403)

    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    return send_from_directory(user_dir, filename)


# =========================
# Pages (메인/메뉴/기능)
# =========================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    gate = login_required_or_redirect()
    if gate:
        return gate
    return render_template("dashboard.html")


@app.route("/planner")
def planner():
    gate = login_required_or_redirect()
    if gate:
        return gate
    return render_template("planner.html")


# ✅ 기능1: 시간표 이미지(여러 장 갤러리)
@app.route("/feature1")
def feature1():
    gate = login_required_or_redirect()
    if gate:
        return gate

    uid = current_user_id()
    conn = db()
    cur = conn.cursor()

    cur.execute("""
      SELECT id, filename, uploaded_at
      FROM user_images
      WHERE user_id=? AND image_key='timetable'
      ORDER BY uploaded_at DESC
    """, (uid,))
    rows = cur.fetchall()
    conn.close()

    images = []
    for r in rows:
        images.append({
            "id": r["id"],
            # ✅ /static/uploads/... 대신 /uploads/...로 서빙 (디스크에서도 유지)
            "url": url_for("uploaded_file", user_id=uid, filename=r["filename"]),
            "uploaded_at": r["uploaded_at"]
        })

    return render_template("feature_timetable.html", images=images)


# 기능2: 공부 기록 & 통계
@app.route("/feature2")
def feature2():
    gate = login_required_or_redirect()
    if gate:
        return gate
    return render_template("feature_stats.html")


# 기능3: 공부 패턴 분석
@app.route("/feature3")
def feature3():
    gate = login_required_or_redirect()
    if gate:
        return gate
    return render_template("feature_pattern.html")


# ✅ 기능4: 학습 회고 노트
@app.route("/feature4")
def feature4():
    gate = login_required_or_redirect()
    if gate:
        return gate
    return render_template("feature_review.html")


# ✅ 기능5: 집중 타이머
@app.route("/feature5")
def feature5():
    gate = login_required_or_redirect()
    if gate:
        return gate
    return render_template("feature_timer.html")


# =========================
# Auth (회원가입/로그인/로그아웃)
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    password2 = request.form.get("password2", "")

    if len(username) < 3:
        flash("아이디는 3자 이상으로 해주세요.")
        return redirect(url_for("register"))
    if len(password) < 6:
        flash("비밀번호는 6자 이상으로 해주세요.")
        return redirect(url_for("register"))
    if password != password2:
        flash("비밀번호가 일치하지 않습니다.")
        return redirect(url_for("register"))

    conn = db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users(username, pw_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        flash("이미 존재하는 아이디입니다.")
        return redirect(url_for("register"))
    finally:
        conn.close()

    flash("회원가입 완료! 로그인해 주세요.")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cur.fetchone()
    conn.close()

    if not user or not check_password_hash(user["pw_hash"], password):
        flash("아이디 또는 비밀번호가 올바르지 않습니다.")
        return redirect(url_for("login"))

    session["user_id"] = user["id"]
    session["username"] = user["username"]

    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# =========================
# API: Plans CRUD
# =========================
@app.route("/api/plans", methods=["GET"])
def api_plans_get():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    uid = current_user_id()
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM plans WHERE user_id=? ORDER BY id DESC", (uid,))
    plans = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"ok": True, "plans": plans})


@app.route("/api/plans", methods=["POST"])
def api_plans_add():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    data = request.get_json(force=True) or {}
    subject = (data.get("subject") or "").strip()
    pages = data.get("pages")
    dday = data.get("dday")

    if not subject:
        return jsonify({"ok": False, "error": "subject_required"}), 400

    try:
        pages = int(pages)
        if pages <= 0:
            raise ValueError()
    except Exception:
        return jsonify({"ok": False, "error": "pages_invalid"}), 400

    try:
        parse_yyyy_mm_dd(dday)
    except Exception:
        return jsonify({"ok": False, "error": "dday_invalid"}), 400

    uid = current_user_id()
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO plans(user_id, subject, pages, dday, created_at) VALUES (?, ?, ?, ?, ?)",
        (uid, subject, pages, dday, datetime.now().isoformat(timespec="seconds"))
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return jsonify({"ok": True, "id": new_id})


@app.route("/api/plans/<int:plan_id>", methods=["DELETE"])
def api_plans_delete(plan_id):
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    uid = current_user_id()
    conn = db()
    cur = conn.cursor()

    cur.execute("""
      DELETE FROM daily_logs
      WHERE plan_id IN (SELECT id FROM plans WHERE id=? AND user_id=?)
    """, (plan_id, uid))

    cur.execute("DELETE FROM plans WHERE id=? AND user_id=?", (plan_id, uid))
    conn.commit()
    deleted = cur.rowcount
    conn.close()

    return jsonify({"ok": True, "deleted": deleted})


# =========================
# API: 오늘(목표/이행) + 로그 저장
# =========================
@app.route("/api/today", methods=["GET"])
def api_today():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    uid = current_user_id()
    today = date.today()
    today_s = today.isoformat()

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM plans WHERE user_id=? ORDER BY id DESC", (uid,))
    plans = cur.fetchall()

    result = []
    for p in plans:
        plan_id = p["id"]
        total_pages = p["pages"]
        dday = parse_yyyy_mm_dd(p["dday"])
        days_left = diff_days_inclusive(today, dday)

        cur.execute("SELECT COALESCE(SUM(pages_done), 0) AS s FROM daily_logs WHERE plan_id=?", (plan_id,))
        done_total = cur.fetchone()["s"]

        remaining = max(0, total_pages - done_total)
        target_range = calc_target_range(remaining, days_left)

        cur.execute("SELECT pages_done FROM daily_logs WHERE plan_id=? AND log_date=?", (plan_id, today_s))
        row = cur.fetchone()
        done_today = row["pages_done"] if row else 0

        met = False
        if target_range is not None:
            mn, mx = target_range
            met = (done_today >= mn) if remaining > 0 else True

        result.append({
            "plan_id": plan_id,
            "subject": p["subject"],
            "total_pages": total_pages,
            "dday": p["dday"],
            "days_left": days_left,
            "done_total": done_total,
            "remaining": remaining,
            "target_min": target_range[0] if target_range else None,
            "target_max": target_range[1] if target_range else None,
            "done_today": done_today,
            "met_today": met
        })

    conn.close()
    return jsonify({"ok": True, "today": today_s, "items": result})


@app.route("/api/log", methods=["POST"])
def api_log_upsert():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    data = request.get_json(force=True) or {}
    plan_id = data.get("plan_id")
    pages_done = data.get("pages_done")

    try:
        plan_id = int(plan_id)
    except Exception:
        return jsonify({"ok": False, "error": "plan_id_invalid"}), 400

    try:
        pages_done = int(pages_done)
        if pages_done < 0:
            raise ValueError()
    except Exception:
        return jsonify({"ok": False, "error": "pages_done_invalid"}), 400

    uid = current_user_id()
    today_s = date.today().isoformat()

    conn = db()
    cur = conn.cursor()

    cur.execute("SELECT id FROM plans WHERE id=? AND user_id=?", (plan_id, uid))
    if not cur.fetchone():
        conn.close()
        return jsonify({"ok": False, "error": "not_found"}), 404

    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
      INSERT INTO daily_logs(plan_id, log_date, pages_done, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(plan_id, log_date)
      DO UPDATE SET pages_done=excluded.pages_done, updated_at=excluded.updated_at
    """, (plan_id, today_s, pages_done, now, now))

    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# =========================
# API: 시간표 이미지 업로드(여러 장 누적)
# =========================
@app.route("/api/upload/timetable", methods=["POST"])
def upload_timetable():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    if "file" not in request.files:
        return jsonify({"ok": False, "error": "file_required"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "empty_filename"}), 400

    if not allowed_file(f.filename):
        return jsonify({"ok": False, "error": "invalid_extension"}), 400

    size = get_file_size(f)
    if size > MAX_FILE_SIZE:
        return jsonify({"ok": False, "error": "file_too_large"}), 400

    uid = current_user_id()
    user_dir = ensure_user_upload_dir(uid)

    base = secure_filename(f.filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts}_{base}"

    save_path = os.path.join(user_dir, filename)
    f.save(save_path)

    conn = db()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO user_images(user_id, image_key, filename, uploaded_at)
      VALUES (?, 'timetable', ?, ?)
    """, (uid, filename, datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    conn.close()

    return jsonify({"ok": True})


# =========================
# API: 시간표 이미지 삭제(개별)
# =========================
@app.route("/api/delete/timetable/<int:image_id>", methods=["POST"])
def delete_timetable(image_id):
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    uid = current_user_id()
    conn = db()
    cur = conn.cursor()

    cur.execute("""
      SELECT filename FROM user_images
      WHERE id=? AND user_id=? AND image_key='timetable'
    """, (image_id, uid))
    row = cur.fetchone()

    if not row:
        conn.close()
        return jsonify({"ok": False, "error": "not_found"}), 404

    file_path = os.path.join(UPLOAD_DIR, str(uid), row["filename"])
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception:
        pass

    cur.execute("DELETE FROM user_images WHERE id=? AND user_id=?", (image_id, uid))
    conn.commit()
    conn.close()

    return jsonify({"ok": True})


# =========================
# API: 공부 통계 (기능2)
# =========================
@app.route("/api/stats", methods=["GET"])
def api_stats():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    uid = current_user_id()

    try:
        days = int(request.args.get("days", 120))
        days = max(7, min(days, 365))
    except Exception:
        days = 120

    start_date = (date.today()).toordinal() - (days - 1)
    start_s = date.fromordinal(start_date).isoformat()

    conn = db()
    cur = conn.cursor()

    cur.execute("""
      SELECT dl.log_date AS d, COALESCE(SUM(dl.pages_done), 0) AS v
      FROM daily_logs dl
      JOIN plans p ON p.id = dl.plan_id
      WHERE p.user_id = ?
        AND dl.log_date >= ?
      GROUP BY dl.log_date
      ORDER BY dl.log_date ASC
    """, (uid, start_s))
    by_date = [{"date": r["d"], "value": int(r["v"])} for r in cur.fetchall()]

    cur.execute("""
      SELECT p.subject AS subject, COALESCE(SUM(dl.pages_done), 0) AS total
      FROM plans p
      LEFT JOIN daily_logs dl ON dl.plan_id = p.id
      WHERE p.user_id = ?
      GROUP BY p.id
      ORDER BY total DESC, p.subject ASC
    """, (uid,))
    by_subject = [{"subject": r["subject"], "total": int(r["total"])} for r in cur.fetchall()]

    total_pages_done = sum(x["value"] for x in by_date)
    active_days = sum(1 for x in by_date if x["value"] > 0)

    conn.close()

    return jsonify({
        "ok": True,
        "days": days,
        "start": start_s,
        "end": date.today().isoformat(),
        "total_pages_done": total_pages_done,
        "active_days": active_days,
        "by_date": by_date,
        "by_subject": by_subject
    })


# =========================
# API: 패턴 분석 (기능3)
# =========================
@app.route("/api/pattern", methods=["GET"])
def api_pattern():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    uid = current_user_id()

    try:
        days = int(request.args.get("days", 90))
        days = max(7, min(days, 365))
    except Exception:
        days = 90

    end_d = date.today()
    start_d = end_d.fromordinal(end_d.toordinal() - (days - 1))
    start_s = start_d.isoformat()
    end_s = end_d.isoformat()

    conn = db()
    cur = conn.cursor()

    cur.execute("""
      SELECT dl.log_date AS d, COALESCE(SUM(dl.pages_done), 0) AS v
      FROM daily_logs dl
      JOIN plans p ON p.id = dl.plan_id
      WHERE p.user_id = ?
        AND dl.log_date >= ?
      GROUP BY dl.log_date
      ORDER BY dl.log_date ASC
    """, (uid, start_s))
    rows = cur.fetchall()

    by_date = [{"date": r["d"], "value": int(r["v"])} for r in rows]

    recent_start = end_d.fromordinal(end_d.toordinal() - 13).isoformat()
    cur.execute("""
      SELECT dl.log_date AS d, COALESCE(SUM(dl.pages_done), 0) AS v
      FROM daily_logs dl
      JOIN plans p ON p.id = dl.plan_id
      WHERE p.user_id = ?
        AND dl.log_date >= ?
      GROUP BY dl.log_date
      ORDER BY dl.log_date ASC
    """, (uid, recent_start))
    recent = [{"date": r["d"], "value": int(r["v"])} for r in cur.fetchall()]
    recent_total = sum(x["value"] for x in recent)
    recent_active_days = sum(1 for x in recent if x["value"] > 0)
    recent_avg_active = (recent_total / recent_active_days) if recent_active_days > 0 else 0.0

    wd_sum = [0] * 7
    wd_cnt = [0] * 7
    for x in by_date:
        d0 = date.fromisoformat(x["date"])
        wd = d0.weekday()
        v = x["value"]
        if v > 0:
            wd_sum[wd] += v
            wd_cnt[wd] += 1

    wd_avg = []
    wd_names = ["월", "화", "수", "목", "금", "토", "일"]
    for i in range(7):
        avg = (wd_sum[i] / wd_cnt[i]) if wd_cnt[i] > 0 else 0.0
        wd_avg.append({
            "weekday": wd_names[i],
            "total": wd_sum[i],
            "days": wd_cnt[i],
            "avg_active": round(avg, 2)
        })

    week_map = defaultdict(int)
    for x in by_date:
        d0 = date.fromisoformat(x["date"])
        y, w, _ = d0.isocalendar()
        week_map[(y, w)] += x["value"]

    weekly = []
    tmp = end_d
    for _ in range(8):
        y, w, _ = tmp.isocalendar()
        weekly.append((y, w))
        tmp = tmp.fromordinal(tmp.toordinal() - 7)
    weekly = list(dict.fromkeys(weekly))
    weekly.reverse()

    weekly_series = [{"label": f"{y}-W{w}", "total": int(week_map.get((y, w), 0))} for (y, w) in weekly]

    cur.execute("SELECT id, subject, pages, dday FROM plans WHERE user_id=? ORDER BY id DESC", (uid,))
    plans = cur.fetchall()

    overload_items = []
    for p in plans:
        plan_id = p["id"]
        total_pages = int(p["pages"])
        dday = parse_yyyy_mm_dd(p["dday"])
        days_left = diff_days_inclusive(end_d, dday)

        cur.execute("SELECT COALESCE(SUM(pages_done),0) AS s FROM daily_logs WHERE plan_id=?", (plan_id,))
        done_total = int(cur.fetchone()["s"])
        remaining = max(0, total_pages - done_total)

        req = None if days_left <= 0 else (remaining / days_left)

        overload_items.append({
            "subject": p["subject"],
            "dday": p["dday"],
            "days_left": days_left,
            "remaining": remaining,
            "required_per_day": round(req, 2) if req is not None else None
        })

    conn.close()

    feedback = []

    active_days = sum(1 for x in by_date if x["value"] > 0)
    consistency = active_days / days if days > 0 else 0
    if consistency >= 0.7:
        feedback.append(f"최근 {days}일 중 {active_days}일 공부했어요. 대단한 성실성입니다!")
    elif consistency >= 0.4:
        feedback.append(f"최근 {days}일 중 {active_days}일 공부했어요. 성실하게 잘 진행하고 있어요.")
    else:
        feedback.append(f"최근 {days}일 중 {active_days}일만 공부했어요. 성실성을 올리면 목표 달성이 쉬워져요.")

    best = max(wd_avg, key=lambda x: x["avg_active"])
    worst = min(wd_avg, key=lambda x: x["avg_active"])
    if best["avg_active"] > 0:
        feedback.append(f"공부가 가장 잘 되는 요일은 {best['weekday']}요일(평균 {best['avg_active']}p) 이에요.")
    if worst["avg_active"] == 0:
        feedback.append(f"{worst['weekday']}요일은 기록이 거의 없어요. 그 날엔 10~20분이라도 공부를 시도해 보세요.")

    if recent_avg_active <= 0:
        feedback.append("최근 14일에 기록된 공부량이 거의 없어서, D-day 부담도를 정확히 비교하기 어려워요. 먼저 2~3일만 기록을 쌓아봐요.")
    else:
        risky = []
        for it in overload_items:
            if it["required_per_day"] is None:
                continue
            if it["required_per_day"] >= recent_avg_active * 1.5 and it["remaining"] > 0 and it["days_left"] > 0:
                risky.append(it)

        if risky:
            risky = sorted(risky, key=lambda x: (x["days_left"], -x["required_per_day"]))[:2]
            for r in risky:
                feedback.append(
                    f"⚠️ {r['subject']}은(는) D-{r['days_left']}에 남은 {r['remaining']}p라서 "
                    f"하루 평균 {r['required_per_day']}p가 필요해요. 최근 평균보다 높아 과부하 가능성이 있어요."
                )
        else:
            feedback.append("현재 페이스 기준으로는 과부하 경고가 크지 않아요. (최근 기록 대비)")

    return jsonify({
        "ok": True,
        "days": days,
        "start": start_s,
        "end": end_s,
        "active_days": active_days,
        "recent14_avg_active": round(recent_avg_active, 2),
        "weekday": wd_avg,
        "weekly": weekly_series,
        "overload": overload_items,
        "feedback": feedback
    })


# =========================
# API: 학습 회고 노트 (기능4)
# =========================
@app.route("/api/review", methods=["GET"])
def api_review_get():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    uid = current_user_id()

    q_date = (request.args.get("date") or "").strip()
    q_days = request.args.get("days")

    conn = db()
    cur = conn.cursor()

    # 최근 N일 목록
    if q_days is not None:
        try:
            days = int(q_days)
            days = max(7, min(days, 365))
        except Exception:
            days = 30

        start_ord = date.today().toordinal() - (days - 1)
        start_s = date.fromordinal(start_ord).isoformat()

        cur.execute("""
          SELECT log_date, goal_met, obstacles, note, created_at, updated_at
          FROM reflections
          WHERE user_id=? AND log_date >= ?
          ORDER BY log_date DESC
        """, (uid, start_s))
        rows = cur.fetchall()
        conn.close()

        items = []
        for r in rows:
            items.append({
                "log_date": r["log_date"],
                "goal_met": bool(r["goal_met"]),
                "obstacles": r["obstacles"],
                "note": r["note"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })

        return jsonify({"ok": True, "mode": "list", "days": days, "start": start_s, "end": date.today().isoformat(), "items": items})

    # 단일 조회(기본: 오늘)
    if not q_date:
        q_date = date.today().isoformat()

    try:
        parse_yyyy_mm_dd(q_date)
    except Exception:
        conn.close()
        return jsonify({"ok": False, "error": "date_invalid"}), 400

    cur.execute("""
      SELECT log_date, goal_met, obstacles, note, created_at, updated_at
      FROM reflections
      WHERE user_id=? AND log_date=?
    """, (uid, q_date))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"ok": True, "mode": "single", "item": None})

    item = {
        "log_date": row["log_date"],
        "goal_met": bool(row["goal_met"]),
        "obstacles": row["obstacles"],
        "note": row["note"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    return jsonify({"ok": True, "mode": "single", "item": item})


@app.route("/api/review", methods=["POST"])
def api_review_upsert():
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    uid = current_user_id()
    data = request.get_json(force=True) or {}

    log_date = (data.get("log_date") or "").strip() or date.today().isoformat()
    goal_met = data.get("goal_met", False)
    obstacles = data.get("obstacles", "")
    note = data.get("note", "")

    try:
        parse_yyyy_mm_dd(log_date)
    except Exception:
        return jsonify({"ok": False, "error": "date_invalid"}), 400

    if isinstance(goal_met, bool):
        goal_met_i = 1 if goal_met else 0
    else:
        try:
            goal_met_i = 1 if int(goal_met) == 1 else 0
        except Exception:
            goal_met_i = 0

    if isinstance(obstacles, list):
        obstacles = ",".join([str(x).strip() for x in obstacles if str(x).strip()])
    obstacles = (obstacles or "").strip()
    note = (note or "").strip()

    now = datetime.now().isoformat(timespec="seconds")

    conn = db()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO reflections(user_id, log_date, goal_met, obstacles, note, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(user_id, log_date)
      DO UPDATE SET
        goal_met=excluded.goal_met,
        obstacles=excluded.obstacles,
        note=excluded.note,
        updated_at=excluded.updated_at
    """, (uid, log_date, goal_met_i, obstacles, note, now, now))

    conn.commit()
    conn.close()

    return jsonify({"ok": True})


@app.route("/api/review/<string:log_date>", methods=["DELETE"])
def api_review_delete(log_date):
    if not current_user_id():
        return jsonify({"ok": False, "error": "login_required"}), 401

    try:
        parse_yyyy_mm_dd(log_date)
    except Exception:
        return jsonify({"ok": False, "error": "date_invalid"}), 400

    uid = current_user_id()
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM reflections WHERE user_id=? AND log_date=?", (uid, log_date))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return jsonify({"ok": True, "deleted": deleted})


# =========================
# ✅ 백업(디스크): DB + 업로드를 ZIP으로 저장
#  - /admin/backup/run (GET/POST) : 백업 생성
#  - /admin/backup/list (GET) : 목록
#  - /admin/backup/download/<name> (GET) : 다운로드
#  - 최근 BACKUP_KEEP개만 유지(자동 삭제)
# =========================
def _is_backup_authorized():
    # 1) 로그인되어 있으면 허용
    if current_user_id():
        return True
    # 2) 토큰 허용(선택)
    if BACKUP_TOKEN:
        token = request.args.get("token") or request.headers.get("X-Backup-Token")
        if token and token == BACKUP_TOKEN:
            return True
    return False


def _safe_backup_name(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    return "".join([c for c in (name or "") if c in keep])[:120] or "backup.zip"


def prune_backups(keep: int = 30):
    """BACKUP_DIR 안의 zip 백업을 최근 keep개만 남기고 삭제"""
    try:
        files = []
        for fn in os.listdir(BACKUP_DIR):
            if not fn.lower().endswith(".zip"):
                continue
            p = os.path.join(BACKUP_DIR, fn)
            try:
                st = os.stat(p)
                files.append((st.st_mtime, fn))
            except Exception:
                pass

        files.sort(key=lambda x: x[0], reverse=True)

        for _, fn in files[keep:]:
            try:
                os.remove(os.path.join(BACKUP_DIR, fn))
            except Exception:
                pass
    except Exception:
        pass


def create_backup_zip() -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = _safe_backup_name(f"backup_{ts}.zip")
    fpath = os.path.join(BACKUP_DIR, fname)

    with zipfile.ZipFile(fpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if os.path.isfile(DB):
            z.write(DB, arcname="planner.db")

        if os.path.isdir(UPLOAD_DIR):
            for root, _, files in os.walk(UPLOAD_DIR):
                for fn in files:
                    abs_path = os.path.join(root, fn)
                    rel_path = os.path.relpath(abs_path, DATA_DIR)  # data 기준 상대경로
                    z.write(abs_path, arcname=rel_path)

    prune_backups(keep=BACKUP_KEEP)

    size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
    return {"filename": fname, "bytes": size, "kept": BACKUP_KEEP}


@app.route("/admin/backup/run", methods=["GET", "POST"])
def admin_backup_run():
    if not _is_backup_authorized():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    info = create_backup_zip()
    return jsonify({"ok": True, "backup": info})


@app.route("/admin/backup/list", methods=["GET"])
def admin_backup_list():
    if not _is_backup_authorized():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    items = []
    try:
        for fn in os.listdir(BACKUP_DIR):
            if not fn.lower().endswith(".zip"):
                continue
            p = os.path.join(BACKUP_DIR, fn)
            try:
                st = os.stat(p)
                items.append({
                    "filename": fn,
                    "bytes": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
                })
            except Exception:
                pass
    except Exception:
        pass

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify({"ok": True, "kept": BACKUP_KEEP, "items": items})


@app.route("/admin/backup/download/<path:name>", methods=["GET"])
def admin_backup_download(name):
    if not _is_backup_authorized():
        abort(401)

    safe = _safe_backup_name(name)
    return send_from_directory(BACKUP_DIR, safe, as_attachment=True)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
