import base64
import io
import html
import streamlit as st
import streamlit.components.v1 as components
from .schema import walk_facts

LABELS = {'part_name':'零件名称', 'drawing_no':'图纸编号', 'material':'加工材料',
          'surface_treatment':'表面处理', 'surface_roughness':'表面粗糙度',
          'overall_dimensions.length':'总体长', 'overall_dimensions.width':'总体宽',
          'overall_dimensions.height':'总体高', 'overall_dimensions.unit':'尺寸单位'}
STATUS = {'supported':'有证据', 'needs_review':'待复核', 'missing':'未识别'}

def heading(title, english=''):
    st.markdown(f'<div class="panel-heading">{html.escape(title)}<small>{html.escape(english)}</small></div>', unsafe_allow_html=True)

def viewer(image, height=620):
    preview = image.copy()
    preview.thumbnail((1900,1900))
    buffer = io.BytesIO()
    preview.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode()
    components.html('''<style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;color:#597181}
.bar{height:38px;display:flex;justify-content:space-between;align-items:center;font-size:11px;padding:0 12px;background:#f7fafb;border:1px solid #e0e7ec;border-radius:9px 9px 0 0}
button{background:white;border:1px solid #d6e0e5;border-radius:5px;cursor:pointer;color:#436170;padding:3px 10px;margin-left:4px}
.canvas{height:HEIGHTpx;overflow:auto;background-color:#eaf0f3;background-image:radial-gradient(#cbd7dd .7px,transparent .7px);background-size:16px 16px;border:1px solid #dce5eb;border-top:0;border-radius:0 0 9px 9px;text-align:center;padding:18px}
img{max-width:none;box-shadow:0 5px 25px #32495820;background:white}
</style><div class="bar"><span>图纸预览 · 放大查看细节</span><div><button onclick="zoom(-.25)" aria-label="缩小">−</button><button onclick="reset()">适应</button><button onclick="zoom(.25)" aria-label="放大">＋</button></div></div><div class="canvas" id="c"><img id="drawing" alt="工程图纸预览" src="data:image/png;base64,ENCODED"></div>
<script>let scale=1;const img=document.getElementById('drawing'),c=document.getElementById('c');let fit=1;function paint(){img.style.width=(img.naturalWidth*fit*scale)+'px'}function reset(){scale=1;fit=Math.min((c.clientWidth-40)/img.naturalWidth,(c.clientHeight-40)/img.naturalHeight);paint()}function zoom(v){scale=Math.min(5,Math.max(.5,scale+v));paint()}img.onload=reset;window.onresize=reset;</script>'''.replace('HEIGHT',str(height-45)).replace('ENCODED',encoded), height=height, scrolling=False)

def summary_fields(result):
    for key in ('part_name','drawing_no','material','surface_treatment','surface_roughness'):
        fact = getattr(result, key)
        value = html.escape(fact.value or '—')
        tag = '<span class="mini-tag">待复核</span>' if fact.status == 'needs_review' else ''
        st.markdown(f'<div class="field"><span class="field-label">{LABELS[key]}</span><span class="field-value">{value}{tag}</span></div>', unsafe_allow_html=True)

def fact_rows(result):
    return [{'字段':LABELS.get(path,path), '路径':path, '值':fact.value or '',
             '状态':fact.status, '证据':','.join(fact.evidence_ids), '原文':fact.raw_text, '备注':fact.note}
            for path, fact in walk_facts(result)]
