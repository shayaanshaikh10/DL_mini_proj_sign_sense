param([string]$DocxPath, [string]$JsonPath, [string]$PdfPath)
# Open the report in Word (invisible), read the displayed page number of every
# heading (outline level 1-2) into a JSON file, and optionally export a PDF.
# Used by the report build; see scripts/build_report.py.
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($DocxPath, $false, $true)
    $doc.Repaginate()
    $pages = [ordered]@{}
    foreach ($para in $doc.Paragraphs) {
        if ($para.OutlineLevel -le 2) {
            $text = $para.Range.Text.Trim()
            if ($text -and -not $pages.Contains($text)) {
                # 1 = wdActiveEndAdjustedPageNumber (number as shown in the footer)
                $pages[$text] = $para.Range.Information(1)
            }
        }
    }
    $pages | ConvertTo-Json | Out-File -FilePath $JsonPath -Encoding utf8
    "total pages: " + $doc.ComputeStatistics(2)
    if ($PdfPath) {
        $doc.Fields.Update() | Out-Null
        $doc.ExportAsFixedFormat($PdfPath, 17)
    }
    $doc.Close(0)
} finally {
    $word.Quit()
}
