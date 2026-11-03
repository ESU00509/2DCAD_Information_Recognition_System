import html
import json
from pathlib import Path
import streamlit as st
from PIL import Image
from cad_insight.settings import ROOT, DATA, Settings
from cad_insight.schema import Extraction, Evidence, walk_facts, validate_evidence
from cad_insight.imaging import load_image, overlay
from cad_insight.storage import history, save_revision, csv_bytes
from cad_insight.pipeline import run
from cad_insight.ui import heading, viewer, summary_fields, fact_rows, LABELS, STATUS

st.set_page_config(page_title='图析 · CAD Insight', page_icon='◈', layout='wide')
st.markdown('<style>'+ (ROOT/'assets/style.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)
try:
    settings = Settings.load()
except ValueError:
    st.error('本地 .env 中的超时、请求次数或价格配置不是有效数字，请修正。')
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">◈</span>图析</div><div class="brand-sub">CAD INSIGHT / WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio('工作空间', ['识别工作台','任务档案','实验评测','连接设置'], label_visibility='collapsed')
    st.divider()
    st.caption('运行环境')
    st.markdown('● 本地工作空间')
    st.caption('DeepSeek 已配置' if settings.api_key else 'DeepSeek 等待配置')
    st.markdown('<div class="side-note">让图纸信息清晰可查。<br>本地 OCR · 区域理解 · 证据追溯</div>', unsafe_allow_html=True)

def hero(title, desc):
    state = 'API 已配置' if settings.api_key else '本地就绪 · API 待配置'
    style = '' if settings.api_key else ' wait'
    st.markdown(f'<div class="eyebrow">ENGINEERING DRAWING INTELLIGENCE</div><div class="hero"><div><h1>{title}</h1><p>{desc}</p></div><span class="pill{style}">● {state}</span></div>', unsafe_allow_html=True)

def show_result(record):
    result = Extraction.model_validate(record['extraction'])
    evidence = [Evidence.model_validate(x) for x in record['evidence']]
    facts = list(walk_facts(result))
    filled = sum(f.value is not None for _, f in facts)
    review = sum(f.status == 'needs_review' for _, f in facts)
    st.caption(f"任务 {record['job_id']} · {record['created_at']} · {record['filename']}")
    if record['status'] == 'ocr_only':
        st.info('本地 OCR 结果：仅包含规则可识别字段，尚未经过 DeepSeek 语义提取。')
    elif record['status'] == 'partial':
        st.warning('部分完成：API 尚未完成所有区域，当前结果已保存。重新识别会复用成功区域缓存。')
    cards = st.columns(4)
    for col, label, value in zip(cards, ['已提取字段','待复核字段','处理耗时','API 请求'],
                                 [filled, review, f"{record['elapsed_seconds']:.1f}s", record.get('api',{}).get('requests',0)]):
        col.metric(label, value)
    st.write('')
    left, right = st.columns([1.35,1], gap='large')
    image = Image.open(DATA/'jobs'/record['job_id']/'original.png').convert('RGB')
    with left:
        heading('图纸与证据', 'DRAWING & EVIDENCE')
        options = {'区域总览':[]}
        for path, fact in facts:
            if fact.value:
                options[f'{LABELS.get(path,path)} · {fact.value}'] = fact.evidence_ids
        selected_label = st.selectbox('选择字段定位来源', list(options), key=f"evidence-{record['job_id']}")
        selected = options[selected_label]
        viewer(overlay(image, evidence, selected or None))
        if selected:
            for e in [x for x in evidence if x.id in selected][:4]:
                with st.expander(f'{e.id} · {e.text[:65]}', expanded=True):
                    st.image(image.crop(e.bbox), width='stretch')
                    st.caption(f'原图坐标 {e.bbox}'+(f' · OCR 分数 {e.score:.3f}' if e.score is not None else ' · 图像区域证据'))
    with right:
        tabs = st.tabs(['提取结果','人工复核','OCR 原文'],key=f"tabs-{record['job_id']}")
        with tabs[0]:
            with st.container(border=True):
                heading('零件信息', 'PART PROFILE')
                summary_fields(result)
            with st.container(border=True):
                heading('总体尺寸', 'OVERALL DIMENSIONS')
                cols = st.columns(3)
                for col, label, key in zip(cols,['长','宽','高'],['length','width','height']):
                    col.metric(label, getattr(result.overall_dimensions,key).value or '—')
                st.caption('单位：'+(result.overall_dimensions.unit.value or '未确认')+' · 无明确标注的尺寸保留为空')
            with st.container(border=True):
                heading('特征与参数', 'FEATURES')
                if not result.features:
                    st.caption('暂无可确认的特征；完成模型提取后在此展示。')
                for f in result.features:
                    with st.expander(f'{f.feature_id}  {f.name} · {f.type}'):
                        st.write('数量：'+(f.quantity.value or '未确认'))
                        for key, value in f.parameters.items():
                            st.write(f'{key}：{value.value or "—"}  · {STATUS[value.status]}')
                        st.caption('标注级关联，不代表已确定跨视图几何位置。')
            with st.container(border=True):
                heading('技术要求', 'TECHNICAL NOTES')
                for i, f in enumerate(result.technical_requirements):
                    st.write(f'{i+1}. {f.value}')
                if not result.technical_requirements:
                    st.caption('等待提取；OCR 原文仍可查看。')
        with tabs[1]:
            st.caption('更改字段后保存独立修订稿，自动结果保持原样。证据编号可用逗号分隔。')
            rows = fact_rows(result)
            edits = st.data_editor(rows, hide_index=True, width='stretch', disabled=['字段','路径'],
                                   column_config={'状态':st.column_config.SelectboxColumn(options=['supported','needs_review','missing'])},
                                   key=f"edit-{record['job_id']}")
            if st.button('保存人工修订', key=f"save-{record['job_id']}", type='primary'):
                revised = result.model_copy(deep=True)
                mapping = dict(walk_facts(revised))
                allowed = {e.id for e in evidence}
                try:
                    for row in edits:
                        field = mapping[row['路径']]
                        old = next(r for r in rows if r['路径'] == row['路径'])
                        if row == old:
                            continue
                        ids = [x.strip() for x in str(row['证据'] or '').replace('，',',').split(',') if x.strip()]
                        if not set(ids) <= allowed:
                            raise ValueError('存在未知证据编号，请从左侧或 OCR 原文中选择。')
                        field.value = str(row['值']).strip() if row['值'] is not None and str(row['值']).strip() else None
                        field.status = row['状态'] if field.value else 'missing'
                        field.source, field.evidence_ids = 'human', ids
                        field.raw_text, field.note = str(row['原文'] or ''), str(row['备注'] or '')
                    validate_evidence(revised, evidence)
                    path = save_revision(record['job_id'], revised.model_dump())
                    st.session_state[f"revision_download-{record['job_id']}"] = revised.model_dump_json(indent=2)
                    st.success('修订稿已保存：'+path.name)
                except ValueError as exc:
                    st.error(str(exc))
            if st.session_state.get(f"revision_download-{record['job_id']}"):
                st.download_button('下载最近修订稿', st.session_state[f"revision_download-{record['job_id']}"], 'reviewed.json', 'application/json')
            with st.expander('完整 JSON 修订（可新增特征或技术条款）'):
                raw = st.text_area('编辑完整提取结构', result.model_dump_json(indent=2), height=250, key=f"json-{record['job_id']}")
                if st.button('校验并保存 JSON 修订', key=f"jsonsave-{record['job_id']}"):
                    try:
                        revised = Extraction.model_validate_json(raw)
                        for _, fact in walk_facts(revised):
                            if fact.value: fact.source = 'human'
                        issues = validate_evidence(revised,evidence)
                        path = save_revision(record['job_id'], revised.model_dump())
                        st.success('已保存 '+path.name)
                        if issues: st.warning('部分字段缺少有效证据，已标记待复核。')
                    except ValueError:
                        st.error('JSON 未通过结构校验，请检查字段格式。')
        with tabs[2]:
            texts = [e for e in evidence if e.kind=='ocr']
            st.caption(f'共 {len(texts)} 个文字块'+(' · 命中本地缓存' if record.get('ocr_cached') else ''))
            st.dataframe([{'编号':e.id,'文字':e.text,'分数':round(e.score or 0,3),'坐标':str(e.bbox)} for e in texts], hide_index=True, height=450, width='stretch')
    if record['warnings']:
        with st.expander(f"处理说明 · {len(record['warnings'])} 条", expanded=False):
            for warning in record['warnings']: st.write('• '+warning)
    with st.expander('运行记录与 API 用量'):
        st.json(record.get('api',{}))
        st.dataframe(record['stage_log'], hide_index=True)
    a,b = st.columns(2)
    a.download_button('↓ 导出完整 JSON', json.dumps(record,ensure_ascii=False,indent=2), record['job_id']+'.json', 'application/json', width='stretch')
    b.download_button('↓ 导出字段 CSV', csv_bytes(result), record['job_id']+'.csv', 'text/csv', width='stretch')

if page == '识别工作台':
    hero('从一张图纸，到清晰的结构化信息', '提取尺寸、材料与工艺要求，让每一项结果都有据可查。')
    st.markdown('<div class="step-strip"><span><b>01</b> 选择图纸</span><span>→</span><span><b>02</b> OCR 识别</span><span>→</span><span><b>03</b> 协同提取</span><span>→</span><span><b>04</b> 复核与导出</span></div>', unsafe_allow_html=True)
    left,right = st.columns([1,1.7],gap='large')
    data, filename = None, None
    with left:
        with st.container(border=True):
            heading('开始新的识别', 'NEW ANALYSIS')
            source = st.radio('图纸来源',['本地图纸样例','上传图纸'],horizontal=True)
            if source == '本地图纸样例':
                samples = sorted((ROOT/'2D图纸数据集').glob('*.jpg'))
                names = [p.name for p in samples]
                index = names.index('M2773808G006.jpg') if 'M2773808G006.jpg' in names else 0
                if names:
                    selected = st.selectbox('选择图纸',names,index=index)
                    data,filename = (ROOT/'2D图纸数据集'/selected).read_bytes(),selected
                    st.caption(f'{len(names)} 张本地样例 · JPG 格式')
            else:
                uploaded = st.file_uploader('拖入图纸或点击选择',type=['jpg','jpeg','png'])
                if uploaded: data,filename = uploaded.getvalue(),uploaded.name
            modes = {'本地 OCR · 无需 API':'ocr','大小模型协同 · 推荐':'hybrid','整图 API · 对比实验':'whole','OCR + 文本 API · 降级':'text'}
            selected_mode = st.selectbox('识别方式',list(modes),index=1 if settings.api_key else 0)
            mode = modes[selected_mode]
            st.caption('图像与相关文字将发送到 DeepSeek。' if mode in ('hybrid','whole') else ('仅 OCR 文本与坐标发送到 DeepSeek。' if mode=='text' else '全部在本机处理，不调用大模型。'))
            with st.expander('高级选项'):
                rotated = st.checkbox('增加旋转文字识别（耗时更长）',value=False)
                cache = st.checkbox('复用已完成区域的 API 缓存',value=True)
                custom = st.checkbox('手动调整区域范围',value=False, disabled=mode=='whole')
                manual = None
                if custom and mode!='whole':
                    manual = {}
                    for name,label,default in [('title','标题栏',(27,79,100,100)),('notes','技术要求',(0,72,72,100)),('view_left','主体左侧',(0,0,56,80)),('view_right','主体右侧',(44,0,100,80))]:
                        st.caption(label)
                        x = st.slider(label+' · 横向 %',0,100,(default[0],default[2]),key=name+'x')
                        y = st.slider(label+' · 纵向 %',0,100,(default[1],default[3]),key=name+'y')
                        manual[name] = [x[0]/100,y[0]/100,x[1]/100,y[1]/100]
            disabled = data is None or (mode!='ocr' and not settings.api_key)
            start = st.button('开始识别  →',type='primary',width='stretch',disabled=disabled)
            if mode!='ocr' and not settings.api_key:
                st.caption('请先在「连接设置」按说明配置 API Key，或切换本地 OCR。')
    with right:
        heading('待识别图纸', 'SOURCE DRAWING')
        if data:
            try:
                source_image = load_image(data)
                st.caption(f'{filename}  ·  {source_image.width} × {source_image.height} px')
                if manual:
                    from cad_insight.imaging import locate_regions
                    regions,_ = locate_regions(source_image,[],manual)
                    preview_evidence = [Evidence(id=f'R{i+1:02d}',kind='region',bbox=box,text=name) for i,(name,box) in enumerate(regions)]
                    source_image = overlay(source_image,preview_evidence)
                viewer(source_image,430)
            except Exception as exc:
                st.error('图片无法读取或裁剪范围无效：'+str(exc))
                data = None
        else:
            st.markdown('<div class="empty"><div class="empty-symbol">▧</div><h3>你的下一张图纸，从这里开始</h3><p>上传 JPG / PNG，或选择左侧本地样例<br>识别后可逐项查看原图证据</p></div>',unsafe_allow_html=True)
    if start and data:
        with st.status('正在识别图纸…',expanded=True) as status_box:
            bar = st.progress(0)
            line = st.empty()
            def progress(message,value):
                line.write(message)
                bar.progress(min(1.0,value))
            try:
                record = run(data,filename,mode,settings,progress,manual,rotated,cache)
                st.session_state['active_result'] = record
                status_box.update(label='识别结束 · 结果已保存',state='complete',expanded=False)
            except Exception as exc:
                status_box.update(label='识别未完成',state='error')
                st.error(f'本地处理失败（{type(exc).__name__}）。请检查 OCR 模型是否已下载，或运行诊断命令。')
    if st.session_state.get('active_result'):
        st.divider()
        heading('识别结果', 'ANALYSIS RESULT')
        show_result(st.session_state['active_result'])
elif page == '任务档案':
    hero('每次识别，都有迹可循', '查看历史结果、复核记录和原始图纸，继续未完成的分析。')
    records = history()
    if not records:
        st.info('还没有识别记录。完成第一张图纸后，它会出现在这里。')
    else:
        choice = st.selectbox('选择历史任务',range(len(records)),format_func=lambda i:f"{records[i]['created_at']} · {records[i]['filename']} · {records[i]['mode']}")
        record = records[choice]
        revisions = sorted((DATA/'jobs'/record['job_id']).glob('revision-*.json'),reverse=True)
        if revisions:
            with st.expander(f'人工修订记录 · {len(revisions)} 份'):
                for rev in revisions:
                    st.download_button(rev.stem,rev.read_bytes(),rev.name,'application/json',key=rev.name)
        show_result(record)
elif page == '实验评测':
    hero('用实验检验，而不只看一次结果', '比较 OCR 规则、整图模型与区域协同，记录准确率、覆盖率和调用成本。')
    records = history()
    st.info('准确率需要人工真值。未提供真值前，仅展示真实运行统计，不生成模拟成绩。')
    if records:
        st.dataframe([{'图纸':r['filename'],'模式':r['mode'],'状态':r['status'],'耗时 / 秒':r['elapsed_seconds'],
                       '请求数':r.get('api',{}).get('requests',0),'缓存命中':r.get('api',{}).get('cache_hits',0)} for r in records],hide_index=True,width='stretch')
    heading('三组对照', 'EXPERIMENT DESIGN')
    a,b,c=st.columns(3)
    a.info('A · OCR + 规则\n\n检验基础文字与格式匹配能力。')
    b.info('B · 整图 DeepSeek\n\n检验直接读整图的提取效果。')
    c.info('C · 区域协同\n\n检验 OCR、裁剪与证据校验的共同作用。')
    st.code('.\\.venv\\python.exe -m cad_insight.cli split\n.\\.venv\\python.exe -m cad_insight.cli evaluate --prediction data/jobs/任务编号/result.json --gold 真值.json',language='powershell')
    st.caption('人工复核结果不能直接冒充自动评测结果。详见 README 中的数据标注与实验说明。')
else:
    hero('连接你的 DeepSeek', '应用与 OCR 留在本机，大模型通过 API 提供图纸理解能力。')
    with st.container(border=True):
        heading('连接状态', 'CONNECTION')
        st.write('密钥：'+('已从本地环境读取' if settings.api_key else '尚未配置'))
        st.write('模型：'+settings.model)
        st.write('接口：'+settings.base_url)
        st.caption(f'超时 {settings.timeout:g} 秒 · 单图请求上限 {settings.max_requests} 次 · 含重试与复核')
        st.markdown('在项目根目录将 `.env.example` 复制为 `.env`，填入 API Key 后点击下方刷新。密钥只保存在本机后端配置中。')
        st.code('DEEPSEEK_API_KEY=你的密钥\nDEEPSEEK_BASE_URL=https://api.deepseek.com\nDEEPSEEK_MODEL=deepseek-flash',language='dotenv')
        if st.button('重新读取本地配置'): st.rerun()
        st.caption('配置状态不等于连接已验证。首次真实识别时会检查账户权限、模型能力与余额。')
    with st.container(border=True):
        heading('费用与缓存', 'USAGE CONTROL')
        st.write('默认单图最多 8 次请求；成功区域会缓存，重复运行可复用。')
        st.write('填写 .env 中每百万 token 的输入/输出单价后显示估算费用；未填写时只显示用量。缓存折扣和供应商账单可能造成差异。')
st.markdown('<div class="footer-note">CAD INSIGHT  /  图纸信息识别工作台　·　结果用于辅助识读，数值与工艺请结合原图复核。</div>', unsafe_allow_html=True)
