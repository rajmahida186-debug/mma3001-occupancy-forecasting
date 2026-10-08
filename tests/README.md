# Tests

Test functions written with **pytest**. They check that the code does what it
should, e.g. that cleaning removes frozen sensor readings and that timestamps
survive daylight-saving changes.

Run all tests and produce the HTML test report:

```bash
pytest --html=reports/test_report/report.html --self-contained-html
```
