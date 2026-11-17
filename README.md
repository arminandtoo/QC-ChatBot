# 🎧 چت‌بات پشتیبانی آنلاین با LangGraph

یه چت‌بات پشتیبانی برای فروشگاه آنلاین که جواب‌های مدل زبانی رو با یه منتقد قانون‌مدار چک می‌کنه (حداکثر ۲ بار هم بازنویسیش می‌کنه)
با RAG به سوالا جواب می‌ده سفارش ثبت و لغو و مرجوع می‌کنه و برای کارای حساس از human-in-the-loop کمک می‌گیره تا یه آدم واقعی تاییدشون کنه

## 🔐 کاربرهای نمونه

| نام کاربری | رمز عبور | اسم | نقش | سفارش‌های نمونه |
|---|---|---|---|---|
| `ali` | `ali-1234` | علی محمدی | مشتری | ۱ (در حال پردازش) و ۴ (در انتظار پرداخت) |
| `maryam` | `maryam-1234` | مریم احمدی | مشتری | ۲ (ارسال شده) و ۵ (لغو شده) |
| `mohammad` | `mohammad-1234` | محمد رضایی | مشتری | ۳ (تحویل داده شده و قابل مرجوعی) |
| `operator` | `operator-1234` | اپراتور پشتیبانی | اپراتور | دسترسی به پنل `/operator` |

- هر مشتری فقط سفارش‌های خودش رو می‌بینه و فقط برای همونا می‌تونه درخواست لغو یا مرجوعی بده
  سفارش بقیه هم دقیقاً مثل یه سفارش ناموجود جواب می‌گیره («در حساب کاربری شما پیدا نشد») تا اصلا لو نره همچین سفارشی وجود داره
- سفارش‌های جدید خودکار به اسم همون کاربری که لاگین کرده ثبت می‌شن
- رمزها با `PBKDF2-SHA256` و یه salt جداگانه توی جدول `users` ذخیره می‌شن و به صورت متن ساده نیستن
- پنل اپراتور فقط برای کاربری باز می‌شه که نقشش `operator` باشه

> این رمزها فقط برای دمو و تست هستن توی محیط واقعی حتما باید عوضشون کنی

## 🚀 اجرا

```bash
# PostgreSQL برای checkpointهای LangGraph
docker run -d --name complaints-postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=complaints -p 5432:5432 -v complaints-pgdata:/var/lib/postgresql/data postgres:16-alpine

# پکیج‌ها (توی Python اصلی)
py -3.11 -m pip install langgraph langgraph-checkpoint-postgres "psycopg[binary]" openai python-dotenv gradio numpy
```

فایل `.env`:

```
AVALAI_API_KEY=...
AVALAI_MODEL=gpt-4o-mini
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/complaints
```

```bash
py -3.11 app.py              # رابط وب: http://127.0.0.1:7860  (پنل اپراتور: /operator)
py -3.11 test_solution.py    # تست‌ها (بدون نیاز به API و Postgres)
py -3.11 solution.py --complaint "سفارش ۳ من کجاست؟" --username mohammad --password mohammad-1234
```

دیتابیس فروشگاه (`shop.db` که SQLite هست) توی اولین اجرا خودکار ساخته می‌شه و با دیتای نمونه پر می‌شه اگه خواستی برگردی به دیتای اولیه کافیه همین فایل رو پاک کنی

## 🧪 سناریوهای پیشنهادی برای تست

| کاربر | پیام | نتیجه مورد انتظار |
|---|---|---|
| `maryam` | سفارش ۱ من کجاست؟ | «در حساب کاربری شما پیدا نشد» (چون سفارش ۱ مال علی هست) |
| `maryam` | سفارش ۱ را لغو کن | همون جواب قبلی و هیچ درخواستی برای اپراتور نمی‌ره |
| `ali` | سفارش ۱ را لغو کن | درخواست لغو می‌ره برای تایید اپراتور |
| `mohammad` | سفارش ۳ کجاست؟ ← میخوام مرجوعش کنم | درخواست مرجوعی سفارش ۳ می‌ره برای تایید اپراتور |
| هر مشتری | یه شارژر و یه هدفون چی داری؟ ← جفتشو ثبت کن | خلاصه سفارش میاد و بعدش می‌پرسه «بله / خیر / اصلاح» |
| هر مشتری | شعبه حضوری در شیراز دارید؟ | «نه فقط شعبه اصفهان» (از طریق RAG) |

## 🧭 گراف

```text
START → classify ─┬─ (سفارش بدون ورود) ─→ login_required → END
                  ├─ order_status → END
                  ├─ cancel_order ── interrupt (اپراتور) → END
                  ├─ return_order ── interrupt (اپراتور) → END
                  ├─ build_order → confirm_order ── interrupt (مشتری) ─┬─ بله → save_order → END
                  │                     ↑                              ├─ اصلاح ─┘
                  │                                                    └─ خیر → END
                  ├─ request_callback ── interrupt (اپراتور) → END
                  └─ retrieve → draft ─┬─ (پاسخی نیست) → offer_callback → END
                                       └─ critique ⇄ revise (حداکثر ۲ بار) → finalize ─┬─ approved → END
                                                                                       └─ human_review ── interrupt (اپراتور) → END
```

- وضعیت هر گفتگو با `thread_id` توی PostgreSQL ذخیره می‌شه (`recursion_limit = 15`)
- سفارش‌های ۶ به بعد که قبل از اضافه شدن سیستم لاگین ساخته شدن صاحب ندارن و هیچ مشتری‌ای بهشون دسترسی نداره