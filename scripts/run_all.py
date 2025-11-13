# =========================
# AFTER PIPELINE: GENERATE DAILY + WEEKLY SUMMARY
# =========================
try:
    from summary_compact import write_daily_summary, write_weekly_roundup
except ImportError:
    try:
        from summary import write_daily_summary, write_weekly_roundup
    except ImportError:
        write_daily_summary = None
        write_weekly_roundup = None

if write_daily_summary:
    print("[run_all] Generating daily summary...")
    write_daily_summary()

    # Only run weekly roundup on Sundays (UTC)
    from datetime import datetime
    if datetime.utcnow().weekday() == 6:
        print("[run_all] Generating weekly roundup...")
        write_weekly_roundup()
    else:
        print("[run_all] Skipping weekly roundup (not Sunday)")
else:
    print("[run_all] No summary module found — skipping summaries.")
