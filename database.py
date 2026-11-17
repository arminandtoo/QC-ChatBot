import hashlib
import hmac
import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "shop.db"

NEW_ORDER_STATUS = "در انتظار پرداخت"
CANCELLABLE_STATUSES = (NEW_ORDER_STATUS, "در حال پردازش")
DELIVERED_STATUS = "تحویل داده شده"
RETURN_APPROVED_STATUS = "مرجوعی تأیید شده"
RETURN_WINDOW_DAYS = 7
TIME_FORMAT = "%Y-%m-%d %H:%M"
PASSWORD_ITERATIONS = 200_000

USERS = [
    ("ali", "ali-1234", "علی محمدی", "customer"),
    ("maryam", "maryam-1234", "مریم احمدی", "customer"),
    ("mohammad", "mohammad-1234", "محمد رضایی", "customer"),
    ("operator", "operator-1234", "اپراتور پشتیبانی", "operator"),
]

ORDER_OWNERS = [
    ("ali", 1),
    ("maryam", 2),
    ("mohammad", 3),
    ("ali", 4),
    ("maryam", 5),
]

CATEGORIES = [
    (1, "موبایل", "گوشی‌های هوشمند از برندهای سامسونگ و شیائومی"),
    (2, "لپ‌تاپ", "لپ‌تاپ‌های دانشجویی، اداری و گیمینگ"),
    (3, "لوازم جانبی", "هدفون، شارژر، کابل، پاوربانک و قاب گوشی"),
    (4, "دوره‌های آموزشی", "دوره‌های آنلاین برنامه‌نویسی و علم داده با دسترسی دائمی"),
]

PRODUCTS = [
    (1, "گوشی سامسونگ Galaxy A55", 1, 18500000, 12, "حافظه ۲۵۶ گیگابایت، رم ۸ گیگابایت، گارانتی ۱۸ ماهه"),
    (2, "گوشی شیائومی Redmi Note 13", 1, 11200000, 0, "حافظه ۱۲۸ گیگابایت، رم ۶ گیگابایت، گارانتی ۱۸ ماهه"),
    (3, "لپ‌تاپ لنوو IdeaPad 3", 2, 32000000, 5, "پردازنده Core i5، رم ۱۶ گیگابایت، SSD ۵۱۲ گیگابایت"),
    (4, "لپ‌تاپ ایسوس TUF Gaming F15", 2, 54000000, 3, "پردازنده Core i7، کارت گرافیک RTX 4060، رم ۱۶ گیگابایت"),
    (5, "هدفون بی‌سیم انکر Soundcore Q20", 3, 3400000, 25, "نویز کنسلینگ فعال، ۴۰ ساعت شارژدهی"),
    (6, "شارژر دیواری ۲۵ وات سامسونگ", 3, 950000, 40, "فست شارژ با پورت USB-C"),
    (7, "دوره جامع پایتون", 4, 2900000, 999, "۶۰ ساعت ویدیو، پروژه‌محور، دارای گواهی پایان دوره"),
    (8, "دوره یادگیری ماشین", 4, 4500000, 999, "۸۰ ساعت ویدیو، شامل پروژه‌های واقعی و پشتیبانی مدرس"),
]

ORDERS = [
    (1, "علی محمدی", "در حال پردازش", "2026-11-14 10:20"),
    (2, "مریم احمدی", "ارسال شده", "2026-11-12 16:05"),
    (3, "محمد رضایی", "تحویل داده شده", "2026-11-05 09:40"),
    (4, "زهرا حسینی", "در انتظار پرداخت", "2026-11-16 21:10"),
    (5, "امیر کریمی", "لغو شده", "2026-11-10 13:30"),
]

DELIVERIES = [
    ("2026-11-14 18:30", 3),
]

ORDER_ITEMS = [
    (1, 1, 1),
    (2, 3, 1),
    (3, 5, 2),
    (3, 6, 1),
    (4, 7, 1),
    (5, 4, 1),
]


class OutOfStockError(Exception):
    def __init__(self, product_id):
        super().__init__(f"product {product_id} is out of stock")
        self.product_id = product_id

FAQS = [
    ("شرایط مرجوعی کالا چیست؟", "تا ۷ روز پس از تحویل، در صورت سالم بودن کالا و بسته‌بندی، امکان مرجوعی وجود دارد. برای ثبت مرجوعی کافی است در همین چت بنویسید کدام سفارش را می‌خواهید مرجوع کنید؛ درخواست پس از تأیید کارشناس ثبت می‌شود."),
    ("سفارش چقدر طول می‌کشد تا برسد؟", "سفارش‌های تهران ۱ تا ۲ روز کاری و سایر شهرها ۳ تا ۵ روز کاری پس از ثبت تحویل داده می‌شوند."),
    ("هزینه ارسال چقدر است؟", "ارسال برای سفارش‌های بالای ۵ میلیون تومان رایگان است و برای سایر سفارش‌ها ۸۰ هزار تومان هزینه دارد."),
    ("چه روش‌های پرداختی دارید؟", "پرداخت آنلاین با تمام کارت‌های عضو شتاب و پرداخت در محل (فقط در تهران) امکان‌پذیر است."),
    ("چطور سفارشم را لغو کنم؟", "سفارش‌هایی که هنوز ارسال نشده‌اند قابل لغو هستند. شماره سفارش را در چت بنویسید تا درخواست لغو پس از تأیید کارشناس انجام شود."),
    ("پول از حسابم کم شده ولی سفارش ثبت نشده، چه کنم؟", "در این حالت مبلغ معمولاً تا ۷۲ ساعت توسط بانک به حساب شما برمی‌گردد. اگر برنگشت، شماره پیگیری بانکی را برای پشتیبانی ارسال کنید."),
    ("گارانتی محصولات چگونه است؟", "گوشی‌ها ۱۸ ماه و لپ‌تاپ‌ها ۲۴ ماه گارانتی رسمی دارند. لوازم جانبی ۶ ماه ضمانت سلامت فیزیکی دارند."),
    ("دوره آموزشی باز نمی‌شود، چه کنم؟", "ابتدا از حساب کاربری خارج و دوباره وارد شوید و کش مرورگر را پاک کنید. اگر مشکل ادامه داشت، تیم فنی ظرف ۲۴ ساعت دسترسی شما را بررسی می‌کند."),
    ("ساعت کاری پشتیبانی چیست؟", "پشتیبانی تلفنی شنبه تا چهارشنبه از ساعت ۹ تا ۱۷ پاسخگو است و چت آنلاین به‌صورت شبانه‌روزی فعال است."),
    ("شعبه حضوری دارید؟", "فروشگاه فقط یک شعبه حضوری در اصفهان دارد (خیابان چهارباغ عباسی، مجتمع پارسیان، طبقه دوم) که شنبه تا پنجشنبه از ساعت ۱۰ تا ۲۰ باز است. در سایر شهرها شعبه حضوری نداریم و خرید فقط به‌صورت آنلاین انجام می‌شود."),
    ("به خارج از کشور ارسال دارید؟", "ارسال فقط به تمام شهرهای داخل ایران انجام می‌شود و ارسال به خارج از کشور نداریم."),
]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category_id INTEGER REFERENCES categories(id),
            price INTEGER NOT NULL,
            stock INTEGER NOT NULL,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            delivered_at TEXT,
            user_id INTEGER REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL REFERENCES orders(id),
            product_id INTEGER NOT NULL REFERENCES products(id),
            quantity INTEGER NOT NULL,
            unit_price INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS faqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS callback_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            phone TEXT,
            question TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)

        conn.executemany("INSERT OR REPLACE INTO categories VALUES (?, ?, ?)", CATEGORIES)

        migrate_single_product_orders(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(orders)")]
        for column, column_type in [("delivered_at", "TEXT"), ("user_id", "INTEGER REFERENCES users(id)")]:
            if column not in columns:
                conn.execute(f"ALTER TABLE orders ADD COLUMN {column} {column_type}")

        existing_users = {row[0] for row in conn.execute("SELECT username FROM users")}
        for username, password, full_name, role in USERS:
            if username not in existing_users:
                salt = os.urandom(16)
                conn.execute(
                    "INSERT INTO users (username, full_name, role, password_salt, password_hash) VALUES (?, ?, ?, ?, ?)",
                    (username, full_name, role, salt.hex(), hash_password(password, salt)),
                )

        if conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
            conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", PRODUCTS)
            conn.executemany("INSERT INTO orders (id, customer_name, status, created_at) VALUES (?, ?, ?, ?)", ORDERS)
            conn.executemany(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price) SELECT ?, id, ?, price FROM products WHERE id = ?",
                [(order_id, quantity, product_id) for order_id, product_id, quantity in ORDER_ITEMS],
            )

        conn.executemany("UPDATE orders SET delivered_at = ? WHERE id = ? AND delivered_at IS NULL", DELIVERIES)
        conn.executemany(
            "UPDATE orders SET user_id = (SELECT id FROM users WHERE username = ?) WHERE id = ? AND user_id IS NULL",
            ORDER_OWNERS,
        )

        existing = {row[0] for row in conn.execute("SELECT question FROM faqs")}
        conn.executemany("UPDATE faqs SET answer = ? WHERE question = ?", [(a, q) for q, a in FAQS if q in existing])
        conn.executemany(
            "INSERT INTO faqs (question, answer) VALUES (?, ?)",
            [faq for faq in FAQS if faq[0] not in existing],
        )


def migrate_single_product_orders(conn):
    columns = [row[1] for row in conn.execute("PRAGMA table_info(orders)")]
    if "product_id" not in columns:
        return

    conn.executescript("""
        INSERT INTO order_items (order_id, product_id, quantity, unit_price)
            SELECT o.id, o.product_id, o.quantity, p.price
            FROM orders o JOIN products p ON p.id = o.product_id;
        CREATE TABLE orders_new (
            id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            delivered_at TEXT
        );
        INSERT INTO orders_new (id, customer_name, status, created_at, updated_at)
            SELECT id, customer_name, status, created_at, updated_at FROM orders;
        DROP TABLE orders;
        ALTER TABLE orders_new RENAME TO orders;
    """)


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PASSWORD_ITERATIONS).hex()


def public_user(row):
    return {"id": row["id"], "username": row["username"], "full_name": row["full_name"], "role": row["role"]}


def authenticate(username, password):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return None
    expected = row["password_hash"]
    actual = hash_password(password, bytes.fromhex(row["password_salt"]))
    return public_user(row) if hmac.compare_digest(actual, expected) else None


def get_user(username):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return public_user(row) if row else None


def now():
    return datetime.now().strftime(TIME_FORMAT)


def get_categories():
    with get_connection() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM categories")]


def get_products():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT p.*, c.name AS category
            FROM products p JOIN categories c ON c.id = p.category_id
        """)
        return [dict(row) for row in rows]


def get_faqs():
    with get_connection() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM faqs")]


def get_product(product_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        return dict(row) if row else None


def create_order(user_id, customer_name, items):
    with get_connection() as conn:
        order_id = conn.execute(
            "INSERT INTO orders (user_id, customer_name, status, created_at) VALUES (?, ?, ?, ?)",
            (user_id, customer_name, NEW_ORDER_STATUS, now()),
        ).lastrowid

        for item in items:
            updated = conn.execute(
                "UPDATE products SET stock = stock - ? WHERE id = ? AND stock >= ?",
                (item["quantity"], item["product_id"], item["quantity"]),
            ).rowcount
            if not updated:
                raise OutOfStockError(item["product_id"])
            conn.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price) SELECT ?, id, ?, price FROM products WHERE id = ?",
                (order_id, item["quantity"], item["product_id"]),
            )
        return order_id


def get_order_items(conn, order_id):
    rows = conn.execute("""
        SELECT i.product_id, p.name AS product_name, i.quantity, i.unit_price
        FROM order_items i JOIN products p ON p.id = i.product_id
        WHERE i.order_id = ?
        ORDER BY i.id
    """, (order_id,))
    return [dict(row) for row in rows]


def get_order(order_id):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if not row:
            return None
        items = get_order_items(conn, order_id)
        return {
            **dict(row),
            "items": items,
            "total": sum(item["quantity"] * item["unit_price"] for item in items),
        }


def get_orders():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT o.id, COALESCE(u.username, '-') AS owner, o.customer_name,
                   GROUP_CONCAT(p.name || ' × ' || i.quantity, '، ') AS items,
                   SUM(i.quantity * i.unit_price) AS total,
                   o.status, o.created_at, o.updated_at
            FROM orders o
            LEFT JOIN users u ON u.id = o.user_id
            JOIN order_items i ON i.order_id = o.id
            JOIN products p ON p.id = i.product_id
            GROUP BY o.id
            ORDER BY o.id
        """)
        return [dict(row) for row in rows]


def cancel_order(order_id):
    with get_connection() as conn:
        conn.execute("UPDATE orders SET status = 'لغو شده', updated_at = ? WHERE id = ?", (now(), order_id))
        conn.execute("""
            UPDATE products
            SET stock = stock + (SELECT SUM(quantity) FROM order_items WHERE order_id = ? AND product_id = products.id)
            WHERE id IN (SELECT product_id FROM order_items WHERE order_id = ?)
        """, (order_id, order_id))


def approve_return(order_id):
    with get_connection() as conn:
        conn.execute("UPDATE orders SET status = ?, updated_at = ? WHERE id = ?", (RETURN_APPROVED_STATUS, now(), order_id))


def create_callback_request(thread_id, phone, question):
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO callback_requests (thread_id, phone, question, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (thread_id, phone, question, "ثبت شده", now()),
        )
        return cursor.lastrowid


def get_callback_requests():
    with get_connection() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM callback_requests ORDER BY id DESC")]


init_db()
