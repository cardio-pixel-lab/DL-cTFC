#!/usr/bin/env node
/*
 * QC 页面（qc_report.html）内联 JS 的无浏览器自检。
 *
 * 用最小 DOM stub 真实执行页面脚本，验证：卡片渲染、优先级筛选、审核勾选、
 * localStorage 落盘、待审计数、导出 CSV 的字段与转义、点击放大开合。
 * 这些是纯语法检查（node --check）覆盖不到的运行时绑定错误——一旦出错，
 * 页面会整体失效（筛选/审核/导出全不可用）。
 *
 * 用法（先生成页面）:
 *   python main.py -r <队列> -o <输出> --qc --no-visualize ...
 *   node test_qc_report.js <输出目录>/qc/qc_report.html
 */

'use strict';

const fs = require('fs');
const vm = require('vm');

const htmlPath = process.argv[2];
if (!htmlPath) {
  console.error('用法: node test_qc_report.js <qc_report.html>');
  process.exit(2);
}
const html = fs.readFileSync(htmlPath, 'utf8');
const jsMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!jsMatch) {
  console.error('未在 HTML 中找到 <script> 块');
  process.exit(2);
}
const js = jsMatch[1];

// ---------------------------------------------------------------- 最小 DOM

function makeEl(init = {}) {
  return {
    tag: init.tag || 'div',
    dataset: init.dataset || {},
    value: init.value,
    checked: false,
    src: init.src,
    textContent: '',
    hidden: false,
    _listeners: {},
    classList: {
      _s: new Set(init.classes || []),
      add(c) { this._s.add(c); },
      remove(...cs) { cs.forEach(c => this._s.delete(c)); },
      contains(c) { return this._s.has(c); },
      toggle(c, force) {
        const on = force === undefined ? !this._s.has(c) : !!force;
        on ? this._s.add(c) : this._s.delete(c);
        return on;
      },
    },
    addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); },
    dispatch(ev, extra = {}) { (this._listeners[ev] || []).forEach(fn => fn({ target: this, ...extra })); },
    getBoundingClientRect() { return { top: 100 }; },
    scrollIntoView() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    appendChild() {},
    click() { this._clicked = true; },
  };
}

const cardBlocks = html.match(/<section class="card"[\s\S]*?<\/section>/g) || [];
if (!cardBlocks.length) {
  console.error('页面中没有任何 .card 分支卡片');
  process.exit(2);
}

const cards = cardBlocks.map(block => {
  const ds = {};
  block.replace(/data-([\w-]+)="([^"]*)"/g, (_, k, v) => {
    ds[k.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = v;
    return '';
  });
  const card = makeEl({ tag: 'section', dataset: ds });
  card._radios = [...block.matchAll(/<input type="radio" name="[^"]*" value="([^"]+)"/g)]
    .map(m => makeEl({ tag: 'input', value: m[1] }));
  card._ta = makeEl({ tag: 'textarea' });
  card.querySelectorAll = sel => (sel.includes('radio') ? card._radios : []);
  card.querySelector = sel => {
    if (sel === 'textarea') return card._ta;
    if (sel.includes('radio') && sel.includes(':checked')) {
      return card._radios.find(r => r.checked) || null;
    }
    return null;
  };
  return card;
});

const imgs = [...html.matchAll(/<img src="([^"]+)" data-cap="([^"]*)"/g)]
  .map(m => makeEl({ tag: 'img', src: m[1], dataset: { cap: m[2] } }));

const filterBtns = [...html.matchAll(/<button data-filter="([^"]+)"/g)]
  .map(m => makeEl({ tag: 'button', dataset: { filter: m[1] } }));

const ids = {};
for (const id of ['modal', 'modal-img', 'modal-cap', 'zoom-btn', 'close-btn', 'export', 'q', 'counts']) {
  if (!html.includes(`id="${id}"`)) {
    console.error(`页面缺少必需元素 id="${id}"`);
    process.exit(2);
  }
  ids[id] = makeEl({ tag: 'div' });
}
ids.q.value = '';

let lastBlob = null;
function Blob(parts) { this.parts = parts; lastBlob = this; }

const document = {
  querySelectorAll(sel) {
    if (sel === '.card') return cards;
    if (sel === '.cell img') return imgs;
    if (sel.includes('data-filter')) return filterBtns;
    return [];
  },
  getElementById: id => ids[id],
  addEventListener() {},
  createElement: () => makeEl({ tag: 'a' }),
};

const mem = {};
const localStorage = {
  getItem: k => (k in mem ? mem[k] : null),
  setItem: (k, v) => { mem[k] = v; },
};

const sandbox = {
  document, localStorage, Blob, console, JSON, Array, Object, String,
  setTimeout, clearTimeout,
  URL: { createObjectURL: () => 'blob:test', revokeObjectURL() {} },
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(js, sandbox);

// ---------------------------------------------------------------- 断言

let fail = 0;
let total = 0;
const check = (name, cond, extra = '') => {
  total++;
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? '   ' + extra : ''}`);
  if (!cond) fail++;
};

const visible = () => cards.filter(c => !c.hidden);
const flagsOf = c => (c.dataset.flags || '').split(',').filter(Boolean);

check('counts 文本已初始化', /共 \d+ 条/.test(ids.counts.textContent), ids.counts.textContent);
check('初始全部可见', visible().length === cards.length, `${visible().length}/${cards.length}`);
check('每卡 3 个审核单选框', cards.every(c => c._radios.length === 3));
check('卡片均有唯一 id', new Set(cards.map(c => c.dataset.id)).size === cards.length);

// 筛选名从页面自身的按钮取（两套 QC 页面的筛选集合不同）
const filterNames = filterBtns.map(b => b.dataset.filter);
check('筛选按钮非空', filterNames.length > 0, filterNames.join(','));
check('筛选含 all / pending', filterNames.includes('all') && filterNames.includes('pending'));

for (const f of filterNames) {
  const btn = filterBtns.find(b => b.dataset.filter === f);
  btn.dispatch('click');
  const expected = f === 'all' ? cards.length
    : f === 'pending' ? cards.length
      : cards.filter(c => flagsOf(c).includes(f)).length;
  check(`筛选 ${f} 可见数正确`, visible().length === expected, `${visible().length} vs ${expected}`);
}

// 审核勾选 + 备注 → localStorage → 计数 → 导出
// 判定值按页面第二档取（两套页面的选项文案不同：通过/存疑/拒绝 vs 掩码可用/掩码不可用/拿不准）
filterBtns.find(b => b.dataset.filter === 'all').dispatch('click');
const c0 = cards[0];
const verdictValue = c0._radios[1].value;
c0._radios.find(r => r.value === verdictValue).dispatch('change');
check('勾选后加 reviewed 类', c0.classList.contains('reviewed'));
check(`勾选后加 v-${verdictValue} 类`, c0.classList.contains('v-' + verdictValue));
check('dataset.verdict 已记录', c0.dataset.verdict === verdictValue, c0.dataset.verdict);
check('localStorage 已写入', Object.keys(mem).length === 1);

c0._ta.value = '近端区含侧支, 存疑';
c0._ta.dispatch('input');

filterBtns.find(b => b.dataset.filter === 'pending').dispatch('click');
check('待审核筛选排除已审', visible().length === cards.length - 1, `${visible().length}`);
check('counts 显示已审核=1', /已审核 1/.test(ids.counts.textContent), ids.counts.textContent);

ids.export.dispatch('click');
check('导出触发 Blob 生成', !!lastBlob);
if (lastBlob) {
  const csv = lastBlob.parts.join('');
  const lines = csv.replace(/^﻿/, '').split('\n');
  check('CSV 行数 = 表头 + 卡片数', lines.length === cards.length + 1, `${lines.length}`);
  check('CSV 表头含审核列', lines[0].includes('审核结论') && lines[0].includes('复核理由'), lines[0]);
  check('CSV 含勾选结论', lines[1].includes(verdictValue), lines[1]);
  check('CSV 引号配对（转义正确）', (csv.match(/"/g) || []).length % 2 === 0);
  check('CSV 列数与表头一致',
    lines.slice(0, 3).every(l => l.split(',').length === lines[0].split(',').length));
}

// 点击放大
imgs[0].dispatch('click');
check('点击帧打开大图', ids.modal.hidden === false);
check('大图标题含帧信息', /f=\d+/.test(ids['modal-cap'].textContent), ids['modal-cap'].textContent);

// 备注写入是 300ms 防抖，等一拍再断言缓存
setTimeout(() => {
  const key = Object.keys(mem)[0];
  check('备注防抖已写入 localStorage', !!key && /侧支/.test(mem[key]));
  console.log('='.repeat(56));
  console.log(fail ? `${fail} 项失败（共 ${total} 项）` : `全部 ${total} 项通过 ✅`);
  process.exit(fail ? 1 : 0);
}, 400);
