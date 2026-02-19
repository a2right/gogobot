# tests/evaluation/test_cases.py
"""
GoGoBot 评估测试用例集
包含：✅ 应该通过的用例 + ❌ 故意设计会失败的用例
运行：python3 tests/evaluation/test_cases.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.evaluation.evaluator import evaluate_plan, print_report, batch_evaluate, summary_stats, print_summary

# ─────────────────────────────────────────────────────────────
# ✅ 应该全部通过的正常用例
# ─────────────────────────────────────────────────────────────
GOOD_CASES = [
    {
        "name": "正常3天行程+预算",
        "query": "规划三天新加坡行程，预算1500新元",
        "itinerary": {
            "days": [
                {"date": "D1", "stops": [
                    {"id":"s1","name":"Gardens by the Bay","zone":"Central",
                     "lat":1.2816,"lng":103.8636,"start":"09:00","end":"12:00",
                     "cost_estimate": 28},
                    {"id":"s2","name":"Marina Bay Sands","zone":"Central",
                     "lat":1.2834,"lng":103.8607,"start":"13:00","end":"17:00",
                     "cost_estimate": 50},
                ]},
                {"date": "D2", "stops": [
                    {"id":"s3","name":"Sentosa Island","zone":"Sentosa",
                     "lat":1.2494,"lng":103.8303,"start":"09:00","end":"13:00",
                     "cost_estimate": 80},
                    {"id":"s4","name":"Universal Studios","zone":"Sentosa",
                     "lat":1.2540,"lng":103.8238,"start":"14:00","end":"18:00",
                     "cost_estimate": 108},
                ]},
                {"date": "D3", "stops": [
                    {"id":"s5","name":"Chinatown","zone":"Chinatown",
                     "lat":1.2838,"lng":103.8445,"start":"10:00","end":"13:00",
                     "cost_estimate": 20},
                    {"id":"s6","name":"Little India","zone":"Little India",
                     "lat":1.3066,"lng":103.8518,"start":"14:00","end":"17:00",
                     "cost_estimate": 15},
                ]},
            ],
            "notes": "总费用约301新元，在预算内"
        },
        "expect_pass": True,
    },
]

# ─────────────────────────────────────────────────────────────
# ❌ 故意设计会失败的用例（验证评估器真的在检测）
# ─────────────────────────────────────────────────────────────
BAD_CASES = [
    {
        "name": "❌ 时间冲突",
        "query": "1天新加坡行程",
        "itinerary": {
            "days": [{"date": "D1", "stops": [
                {"id":"s1","name":"Gardens by the Bay","zone":"Central",
                 "lat":1.2816,"lng":103.8636,
                 "start":"09:00","end":"12:00"},  # 09-12
                {"id":"s2","name":"Marina Bay Sands","zone":"Central",
                 "lat":1.2834,"lng":103.8607,
                 "start":"11:00","end":"14:00"},  # ← 11点开始，与上面冲突
            ]}]
        },
        "expect_pass": False,
        "expect_fail_on": "no_time_conflict",
    },
    {
        "name": "❌ 重复景点",
        "query": "2天新加坡行程",
        "itinerary": {
            "days": [
                {"date": "D1", "stops": [
                    {"id":"s1","name":"Gardens by the Bay","zone":"Central",
                     "lat":1.2816,"lng":103.8636,"start":"09:00","end":"12:00"},
                ]},
                {"date": "D2", "stops": [
                    {"id":"s2","name":"Gardens by the Bay","zone":"Central",  # ← 重复
                     "lat":1.2816,"lng":103.8636,"start":"09:00","end":"12:00"},
                ]},
            ]
        },
        "expect_pass": False,
        "expect_fail_on": "diverse_stops",
    },
    {
        "name": "❌ 超出预算",
        "query": "1天新加坡行程，预算50新元",
        "itinerary": {
            "days": [{"date": "D1", "stops": [
                {"id":"s1","name":"Universal Studios","zone":"Sentosa",
                 "lat":1.2540,"lng":103.8238,
                 "start":"09:00","end":"18:00",
                 "cost_estimate": 108},  # ← 108 > 50
            ]}]
        },
        "expect_pass": False,
        "expect_fail_on": "budget",
    },
    {
        "name": "❌ 坐标超出新加坡范围",
        "query": "1天行程",
        "itinerary": {
            "days": [{"date": "D1", "stops": [
                {"id":"s1","name":"Fake Place","zone":"Central",
                 "lat": 35.6762,   # ← 东京坐标！
                 "lng": 139.6503,
                 "start":"09:00","end":"12:00"},
            ]}]
        },
        "expect_pass": False,
        "expect_fail_on": "valid_coordinates",
    },
    {
        "name": "❌ 空天行程",
        "query": "2天新加坡行程",
        "itinerary": {
            "days": [
                {"date": "D1", "stops": [
                    {"id":"s1","name":"Chinatown","zone":"Chinatown",
                     "lat":1.2838,"lng":103.8445,"start":"10:00","end":"13:00"},
                ]},
                {"date": "D2", "stops": []},  # ← 第2天没有任何景点
            ]
        },
        "expect_pass": False,
        "expect_fail_on": "complete_information",
    },
    {
        "name": "❌ 单天跨越过多区域",
        "query": "1天新加坡行程",
        "itinerary": {
            "days": [{"date": "D1", "stops": [
                {"id":"s1","name":"Changi Airport","zone":"East",
                 "lat":1.3644,"lng":103.9915,"start":"08:00","end":"09:30"},
                {"id":"s2","name":"Jurong Bird Park","zone":"West",
                 "lat":1.3191,"lng":103.7067,"start":"10:00","end":"12:00"},
                {"id":"s3","name":"Little India","zone":"North",
                 "lat":1.3066,"lng":103.8518,"start":"13:00","end":"14:30"},
                {"id":"s4","name":"Sentosa","zone":"South",
                 "lat":1.2494,"lng":103.8303,"start":"15:00","end":"17:00"},
            ]}]
        },
        "expect_pass": False,
        "expect_fail_on": "within_zone",
    },
    {
        "name": "❌ 完全无效的行程",
        "query": "随便来个行程",
        "itinerary": {"_invalid": True},
        "expect_pass": False,
        "expect_fail_on": "delivered",
    },
]


# ─────────────────────────────────────────────────────────────
# 运行测试
# ─────────────────────────────────────────────────────────────
def run_tests():
    all_cases = GOOD_CASES + BAD_CASES
    passed_meta = 0
    failed_meta = 0

    print("\n" + "═"*60)
    print("  GoGoBot 评估器自检（含故意失败用例）")
    print("═"*60)

    for case in all_cases:
        result = evaluate_plan(case["itinerary"], case["query"])
        expect = case["expect_pass"]
        actual = result.final_pass

        # 验证是否符合预期
        meta_ok = (actual == expect)

        # 如果预期失败，额外检查是否失败在正确的约束上
        if not expect and case.get("expect_fail_on"):
            target = case["expect_fail_on"]
            if target == "delivered":
                meta_ok = meta_ok and (not result.delivered)
            else:
                all_constraints = result.commonsense_results + result.hard_results
                target_result = next(
                    (c for c in all_constraints if c.name == target), None
                )
                if target_result:
                    meta_ok = meta_ok and (not target_result.passed)

        status = "✅ PASS" if meta_ok else "❌ FAIL"
        if meta_ok:
            passed_meta += 1
        else:
            failed_meta += 1

        print(f"\n[{status}] {case['name']}")
        print(f"  期望final_pass={expect}, 实际={actual}")

        # 打印失败的约束
        if not result.delivered:
            print("  → 未能生成有效行程")
        else:
            failed_cs = [c for c in result.commonsense_results if not c.passed]
            failed_hc = [c for c in result.hard_results if not c.passed]
            if failed_cs:
                print(f"  → 失败的常识约束: {[c.name for c in failed_cs]}")
            if failed_hc:
                print(f"  → 失败的硬约束:   {[c.name for c in failed_hc]}")

    # 汇总
    total = len(all_cases)
    print("\n" + "═"*60)
    print(f"  自检结果: {passed_meta}/{total} 用例符合预期")
    if failed_meta == 0:
        print("  ✅ 评估器工作正常！所有用例行为符合预期。")
    else:
        print(f"  ⚠️  {failed_meta} 个用例行为不符合预期，评估器可能有问题。")
    print("═"*60)

    # 单独对 BAD_CASES 跑一次 batch summary，验证指标不是全绿
    print("\n  [Bad Cases 单独汇总] — 指标应该 < 100%：")
    bad_plans = [{"itinerary": c["itinerary"], "query": c["query"]} for c in BAD_CASES]
    bad_results = batch_evaluate(bad_plans)
    bad_stats = summary_stats(bad_results)
    print_summary(bad_stats)


if __name__ == "__main__":
    run_tests()