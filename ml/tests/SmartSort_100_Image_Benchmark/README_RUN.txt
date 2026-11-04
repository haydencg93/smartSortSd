Smart Sort 100-image benchmark compatibility update
====================================================

Only benchmark.py is a replacement project file. The ML model, setup_data.py,
train_data.py, app.py, training folders, and original test photos are untouched.

BACK UP AND REPLACE:
  cd D:\uiSmartSort\ml\tests
  Copy-Item .\SmartSort_100_Image_Benchmark\benchmark.py .\SmartSort_100_Image_Benchmark\benchmark_before_11class.py
  [Extract this ZIP to D:\uiSmartSort\ml\tests\SmartSort_100_Image_Benchmark, replacing benchmark.py]

RUN:
  python .\SmartSort_100_Image_Benchmark\benchmark.py `
    --manifest ".\heldout_neoavara\manifest.csv" `
    --images ".\heldout_neoavara\images" `
    --checkpoint "..\smart_sort_mobilenetv2_combined.pth" `
    --output ".\results_100_combined"

OPEN:
  Start-Process ".\results_100_combined\SmartSort_100_Image_Report.html"

CONDITIONS:
  - Requires existing 100 benchmark photographs and manifest.
  - Class mapping is auto-loaded from .pth adjacent .classes.json.
  - Only six material categories are present in the benchmark's original manifest.
  - 'trash' and 'landfill' are normalized as BIN NAMES for comparison.
  - Bin expected values reflect historic provisional Iowa City routing;
    NOT item-reviewed University of Iowa bin rules.
  - Source photos may overlap training data. Check provenance separately.
