"""Unit tests for CodeVerifier."""

from aarohan.tools.code_verifier import CodeVerifier


def test_code_verifier_python():
    v = CodeVerifier()
    res = v.verify_python("def add(a, b):\n    return a + b\n")
    assert res["valid"] is True
    assert res["functions"] == 1

    bad = v.verify_python("def bad(: return 1")
    assert bad["valid"] is False
    assert "SyntaxError" in bad["error"]


def test_code_verifier_html():
    v = CodeVerifier()
    valid_html = (
        "<!DOCTYPE html>\n<html>\n<head><title>App</title><style>body{color:#fff}</style></head>\n"
        "<body><h1>Hello World</h1><script>console.log('hi');</script></body>\n</html>"
    )
    res = v.verify_html_website(valid_html)
    assert res["valid"] is True
    assert res["has_scripts"] is True
    assert res["has_styles"] is True


def test_code_verifier_sql():
    v = CodeVerifier()
    valid_sql = "SELECT department, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*) > 2;"
    res = v.verify_sql(valid_sql)
    assert res["valid"] is True
    assert res["dialect"] in ("SQLite", "Standard SQL")


def test_code_verifier_javascript():
    v = CodeVerifier()
    valid_js = "function calculateSum(arr) { return arr.reduce((acc, x) => acc + x, 0); }"
    res = v.verify_javascript(valid_js)
    assert res["valid"] is True

    bad_js = "function calculateSum(arr) { return arr.reduce((acc, x) => acc + x, 0;"
    bad_res = v.verify_javascript(bad_js)
    assert bad_res["valid"] is False


def test_code_verifier_dataset():
    v = CodeVerifier()
    valid_json = '[{"id": 1, "name": "Aarohan"}, {"id": 2, "name": "Studio"}]'
    res = v.verify_data(valid_json, "json")
    assert res["valid"] is True
    assert res["count"] == 2

    valid_csv = "id,name,role\n1,Datta,Lead\n2,Aarohan,AI"
    csv_res = v.verify_data(valid_csv, "csv")
    assert csv_res["valid"] is True
    assert csv_res["rows"] == 3
    assert csv_res["columns"] == 3
