import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'
OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    def settle():
        page.wait_for_function("!document.querySelector('[data-stale=\"true\"]')",timeout=30000)
        page.wait_for_timeout(600)
    page.goto('http://127.0.0.1:8502',wait_until='domcontentloaded')
    page.get_by_role('button',name='开始识别').wait_for(timeout=90000)
    settle()
    page.frame_locator('iframe').first.locator('#drawing').wait_for(timeout=30000)
    page.frame_locator('iframe').first.locator('#drawing').evaluate('(img) => img.decode()')
    page.screenshot(path=str(OUT/'workbench.png'),full_page=True)
    page.get_by_text('任务档案',exact=True).click()
    page.get_by_text('每次识别，都有迹可循',exact=True).wait_for(timeout=30000)
    page.screenshot(path=str(OUT/'history.png'),full_page=True)
    page.get_by_text('连接设置',exact=True).click()
    page.get_by_text('连接你的 DeepSeek',exact=True).wait_for(timeout=30000)
    page.screenshot(path=str(OUT/'settings.png'),full_page=True)
    page.get_by_text('识别工作台',exact=True).click()
    page.get_by_role('button',name='开始识别').wait_for()
    page.get_by_role('button',name='开始识别').click()
    page.get_by_text('本地 OCR 结果：仅包含规则可识别字段，尚未经过 DeepSeek 语义提取。',exact=True).wait_for(timeout=180000)
    page.get_by_role('button',name='↓ 导出完整 JSON').wait_for(timeout=30000)
    page.get_by_text('人工复核',exact=True).click()
    page.get_by_role('button',name='保存人工修订',exact=True).click()
    page.get_by_text('修订稿已保存：',exact=False).wait_for(timeout=15000)
    with page.expect_download() as download:
        page.get_by_role('button',name='↓ 导出完整 JSON').click()
    download.value.save_as(str(OUT/'download-check.json'))
    downloaded=json.loads((OUT/'download-check.json').read_text(encoding='utf-8'))
    assert downloaded['status']=='ocr_only'
    page.get_by_text('提取结果',exact=True).click()
    settle()
    page.evaluate("document.querySelector('[data-testid=stMain]').scrollTop=0")
    page.screenshot(path=str(OUT/'result.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    page.reload(wait_until='domcontentloaded')
    page.get_by_role('button',name='开始识别').wait_for(timeout=30000)
    settle()
    close_sidebar=page.locator('[data-testid="stSidebarCollapseButton"] button')
    if close_sidebar.is_visible():
        close_sidebar.click()
    settle()
    page.screenshot(path=str(OUT/'mobile.png'),full_page=True)
    assert page.locator('[data-testid="stException"]').count()==0
    (OUT/'browser-check.json').write_text(json.dumps({'page_errors':errors,'download_job':downloaded['job_id'],
                            'checked':['workbench','history','settings','real OCR','revision save','JSON download','mobile']},ensure_ascii=False,indent=2),encoding='utf-8')
    browser.close()
    if errors:raise RuntimeError(errors)
    print('Browser checks passed')
