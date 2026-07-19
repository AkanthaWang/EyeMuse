from __future__ import annotations

from html import escape
from typing import Any


def build_dashboard_chart_html() -> str:
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: transparent;
      font-family: Inter, "Segoe UI", "Microsoft YaHei", sans-serif;
    }
    body {
      background:
        radial-gradient(circle at 14% 12%, rgba(166, 239, 255, 0.58), transparent 26%),
        radial-gradient(circle at 86% 10%, rgba(173, 187, 255, 0.38), transparent 22%),
        linear-gradient(180deg, rgba(237, 249, 255, 0.98), rgba(217, 238, 255, 0.98), rgba(199, 227, 251, 0.98));
    }
    .board {
      width: 100%;
      height: 100%;
      box-sizing: border-box;
      padding: 20px;
      display: grid;
      grid-template-columns: 1.6fr 1fr 1fr;
      grid-template-rows: 1.2fr 1fr 92px;
      gap: 18px;
    }
    .metric-card, .chart-card {
      background:
        linear-gradient(180deg, rgba(233, 246, 255, 0.64), rgba(198, 224, 255, 0.50));
      border: 1px solid rgba(255, 255, 255, 0.84);
      border-radius: 18px;
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.48),
        0 10px 28px rgba(131, 176, 228, 0.18);
    }
    .chart-card {
      position: relative;
      padding: 14px;
    }
    .chart-title {
      position: absolute;
      left: 18px;
      top: 14px;
      z-index: 2;
      color: #2F5C98;
      font-size: 15px;
      font-weight: 600;
    }
    .chart {
      width: 100%;
      height: 100%;
    }
    .chart-status {
      position: absolute;
      inset: 14px;
      z-index: 5;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 14px;
      background: rgba(220, 239, 255, 0.76);
      color: #3B679E;
      font-size: 14px;
    }
    .chart-status.hidden { opacity: 0; pointer-events: none; }
    .metric-row {
      grid-column: 1 / span 3;
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
    }
    .metric-card {
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .metric-label {
      color: #6E8DB8;
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric-value {
      color: #355E96;
      font-size: 28px;
      font-weight: 700;
      line-height: 1;
    }
    .metric-sub {
      color: #6E8DB8;
      font-size: 12px;
      margin-top: 6px;
    }
    .trend { grid-column: 1 / span 2; grid-row: 1; }
    .donut { grid-column: 3; grid-row: 1; }
    .bar { grid-column: 1; grid-row: 2; }
    .radar { grid-column: 2; grid-row: 2; }
    .mini-line { grid-column: 3; grid-row: 2; }
  </style>
</head>
<body>
  <div class="board">
    <div id="dashboardStatus" class="chart-status">正在准备历史分析大屏...</div>
    <div class="chart-card trend"><div class="chart-title">历史趋势折线</div><div id="trendChart" class="chart"></div></div>
    <div class="chart-card donut"><div class="chart-title">情绪结构环形图</div><div id="pieChart" class="chart"></div></div>
    <div class="chart-card bar"><div class="chart-title">主信号柱状图</div><div id="barChart" class="chart"></div></div>
    <div class="chart-card radar"><div class="chart-title">综合画像雷达图</div><div id="radarChart" class="chart"></div></div>
    <div class="chart-card mini-line"><div class="chart-title">近期波动面积图</div><div id="miniLineChart" class="chart"></div></div>
    <div class="metric-row">
      <div class="metric-card"><div class="metric-label">历史平均压力</div><div id="avgStressValue" class="metric-value">0</div><div id="avgStressSub" class="metric-sub">等待载入</div></div>
      <div class="metric-card"><div class="metric-label">历史平均疲劳</div><div id="avgFatigueValue" class="metric-value">0</div><div id="avgFatigueSub" class="metric-sub">等待载入</div></div>
      <div class="metric-card"><div class="metric-label">历史平均专注</div><div id="avgFocusValue" class="metric-value">0</div><div id="avgFocusSub" class="metric-sub">等待载入</div></div>
      <div class="metric-card"><div class="metric-label">键盘活跃均值</div><div id="avgKeyboardValue" class="metric-value">0</div><div id="avgKeyboardSub" class="metric-sub">等待载入</div></div>
      <div class="metric-card"><div class="metric-label">鼠标活跃均值</div><div id="avgMouseValue" class="metric-value">0</div><div id="avgMouseSub" class="metric-sub">等待载入</div></div>
      <div class="metric-card"><div class="metric-label">历史样本规模</div><div id="sampleCountValue" class="metric-value">0</div><div id="sampleCountSub" class="metric-sub">等待载入</div></div>
    </div>
  </div>
  <script src="vendor/echarts.min.js"></script>
  <script>
    const palette = ['#F3D6A4', '#9FB7F8', '#F4A38D', '#B79EF3', '#8FC6E8', '#F0C17B'];
    const textColor = '#3B679E';
    const axisColor = 'rgba(88, 126, 180, 0.22)';
    const gridColor = 'rgba(88, 126, 180, 0.12)';
    const tooltipStyle = {
      backgroundColor: 'rgba(241, 248, 255, 0.96)',
      borderColor: 'rgba(255, 255, 255, 0.84)',
      borderWidth: 1,
      textStyle: { color: '#2F5C98' },
      extraCssText: 'box-shadow:0 18px 32px rgba(131,176,228,0.20); border-radius:14px;'
    };
    const commonGrid = { left: 42, right: 18, top: 52, bottom: 28 };
    window.dashboardCharts = {};
    window.dashboardReady = false;
    window.pendingDashboardPayload = null;
    function setStatus(message, isError = false) {
      const node = document.getElementById('dashboardStatus');
      node.textContent = message || '';
      node.classList.toggle('hidden', !message);
      node.style.color = isError ? '#B65353' : '#3B679E';
    }
    function variance(series) {
      if (!series || !series.length) return 0;
      const average = series.reduce((sum, item) => sum + item, 0) / series.length;
      return series.reduce((sum, item) => sum + Math.pow(item - average, 2), 0) / series.length;
    }
    function initCharts() {
      if (window.dashboardCharts.trendChart) return;
      window.dashboardCharts.trendChart = echarts.init(document.getElementById('trendChart'), null, { renderer: 'canvas' });
      window.dashboardCharts.pieChart = echarts.init(document.getElementById('pieChart'), null, { renderer: 'canvas' });
      window.dashboardCharts.barChart = echarts.init(document.getElementById('barChart'), null, { renderer: 'canvas' });
      window.dashboardCharts.radarChart = echarts.init(document.getElementById('radarChart'), null, { renderer: 'canvas' });
      window.dashboardCharts.miniLineChart = echarts.init(document.getElementById('miniLineChart'), null, { renderer: 'canvas' });
    }
    function updateMetric(id, value, subtext) {
      document.getElementById(id + 'Value').textContent = value;
      document.getElementById(id + 'Sub').textContent = subtext;
    }
    window.updateDashboard = function (payload) {
      window.pendingDashboardPayload = payload;
      if (!window.dashboardReady || !window.echarts) return;
      const categories = payload.line_categories || [];
      const lineSeries = payload.line_series || {};
      const stressSeries = lineSeries['压力'] || [];
      const fatigueSeries = lineSeries['疲劳'] || [];
      const focusSeries = lineSeries['专注'] || [];
      const emotionDistribution = payload.emotion_distribution || [];
      const signalDistribution = payload.signal_distribution || [];
      const averages = payload.averages || {};
      const recentCategories = categories.slice(-6);
      const recentStress = stressSeries.slice(-6);
      const recentFatigue = fatigueSeries.slice(-6);
      const recentFocus = focusSeries.slice(-6);
      const stabilityScore = Math.max(0, 100 - Math.round(variance(recentFocus) * 0.7));
      const rhythmScore = Math.max(0, 100 - Math.round((variance(recentStress) + variance(recentFatigue)) * 0.25));
      const recoveryScore = Math.max(0, 100 - Math.round(((averages.avg_fatigue || 0) + (averages.avg_stress || 0)) * 0.45));
      updateMetric('avgStress', averages.avg_stress || 0, '平滑压力均值');
      updateMetric('avgFatigue', averages.avg_fatigue || 0, '平滑疲劳均值');
      updateMetric('avgFocus', averages.avg_focus || 0, '专注表现概览');
      updateMetric('avgKeyboard', (averages.avg_keyboard_activity || 0).toFixed(2), '键盘活跃度');
      updateMetric('avgMouse', (averages.avg_mouse_activity || 0).toFixed(2), '鼠标活跃度');
      updateMetric('sampleCount', payload.sample_count || 0, payload.period_label || '历史区间');
      window.dashboardCharts.trendChart.setOption({
        animation: false,
        color: palette,
        tooltip: { ...tooltipStyle, trigger: 'axis' },
        legend: { top: 12, right: 10, textStyle: { color: textColor, fontSize: 11 } },
        grid: commonGrid,
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: categories,
          axisLine: { lineStyle: { color: axisColor } },
          axisTick: { show: false },
          axisLabel: { color: '#8A98B0', fontSize: 11 },
        },
        yAxis: {
          type: 'value',
          max: 100,
          splitLine: { lineStyle: { color: gridColor } },
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: '#8A98B0', fontSize: 11 },
        },
        series: [
          {
            name: '压力',
            type: 'line',
            smooth: true,
            symbol: 'none',
            lineStyle: { width: 2, color: '#9FB7F8' },
            areaStyle: { color: 'rgba(159, 183, 248, 0.14)' },
            data: stressSeries,
          },
          {
            name: '疲劳',
            type: 'line',
            smooth: true,
            symbol: 'none',
            lineStyle: { width: 2, color: '#F3D6A4' },
            areaStyle: { color: 'rgba(243, 214, 164, 0.12)' },
            data: fatigueSeries,
          },
          {
            name: '专注',
            type: 'line',
            smooth: true,
            symbol: 'none',
            lineStyle: { width: 2, color: '#F4A38D' },
            areaStyle: { color: 'rgba(244, 163, 141, 0.10)' },
            data: focusSeries,
          },
        ],
      }, true);
      window.dashboardCharts.pieChart.setOption({
        animation: false,
        tooltip: { ...tooltipStyle, trigger: 'item' },
        legend: { bottom: 6, textStyle: { color: textColor, fontSize: 11 } },
        color: palette,
        series: [{
          type: 'pie',
          radius: ['46%', '71%'],
          center: ['50%', '48%'],
          label: { color: textColor, fontSize: 11, formatter: '{b}\\n{d}%' },
          itemStyle: { borderColor: 'rgba(237,249,255,0.98)', borderWidth: 3 },
          data: emotionDistribution,
        }],
      }, true);
      window.dashboardCharts.barChart.setOption({
        animation: false,
        tooltip: { ...tooltipStyle, trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: commonGrid,
        xAxis: {
          type: 'category',
          data: signalDistribution.map(item => item.name),
          axisLine: { lineStyle: { color: axisColor } },
          axisTick: { show: false },
          axisLabel: { color: '#8A98B0', fontSize: 11 },
        },
        yAxis: {
          type: 'value',
          splitLine: { lineStyle: { color: gridColor } },
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: '#8A98B0', fontSize: 11 },
        },
        series: [{
          type: 'bar',
          barWidth: '28%',
          data: signalDistribution.map(item => item.value),
          itemStyle: {
            borderRadius: [8, 8, 0, 0],
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(244, 216, 170, 0.95)' },
              { offset: 0.55, color: 'rgba(240, 193, 123, 0.88)' },
              { offset: 1, color: 'rgba(244, 163, 141, 0.78)' },
            ]),
          },
        }],
      }, true);
      window.dashboardCharts.radarChart.setOption({
        animation: false,
        radar: {
          center: ['50%', '56%'],
          radius: '62%',
          splitNumber: 5,
          indicator: [
            { name: '稳定', max: 100 },
            { name: '恢复', max: 100 },
            { name: '专注', max: 100 },
            { name: '节奏', max: 100 },
            { name: '低压力', max: 100 },
          ],
          axisName: { color: '#B9C5D8', fontSize: 11 },
          splitLine: { lineStyle: { color: gridColor } },
          splitArea: { areaStyle: { color: ['rgba(24,34,54,0.16)', 'rgba(24,34,54,0.06)'] } },
          axisLine: { lineStyle: { color: axisColor } },
        },
        series: [{
          type: 'radar',
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { width: 1.2, color: '#F3D6A4' },
          areaStyle: { color: 'rgba(243, 214, 164, 0.12)' },
          data: [{
            value: [stabilityScore, recoveryScore, Math.round(averages.avg_focus || 0), rhythmScore, Math.max(0, 100 - Math.round(averages.avg_stress || 0))],
          }],
        }],
      }, true);
      window.dashboardCharts.miniLineChart.setOption({
        animation: false,
        color: palette,
        tooltip: { ...tooltipStyle, trigger: 'axis' },
        legend: { top: 12, textStyle: { color: textColor, fontSize: 11 } },
        grid: commonGrid,
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: recentCategories,
          axisLine: { lineStyle: { color: axisColor } },
          axisTick: { show: false },
          axisLabel: { color: '#8A98B0', fontSize: 10 },
        },
        yAxis: {
          type: 'value',
          max: 100,
          splitLine: { lineStyle: { color: gridColor } },
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: '#8A98B0', fontSize: 10 },
        },
        series: [
          { name: '压力', type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#9FB7F8' }, areaStyle: { color: 'rgba(159, 183, 248, 0.14)' }, data: recentStress },
          { name: '疲劳', type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#F3D6A4' }, areaStyle: { color: 'rgba(243, 214, 164, 0.12)' }, data: recentFatigue },
          { name: '专注', type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#F4A38D' }, areaStyle: { color: 'rgba(244, 163, 141, 0.10)' }, data: recentFocus },
        ],
      }, true);
      setStatus('');
    };
    function bootDashboard() {
      if (!window.echarts) {
        setStatus('本地图表资源加载失败', true);
        return;
      }
      initCharts();
      window.dashboardReady = true;
      if (window.pendingDashboardPayload) {
        const latest = window.pendingDashboardPayload;
        window.pendingDashboardPayload = null;
        window.updateDashboard(latest);
      } else {
        setStatus('正在等待历史数据...');
      }
    }
    window.addEventListener('resize', () => {
      Object.values(window.dashboardCharts).forEach(chart => chart && chart.resize());
    });
    bootDashboard();
  </script>
</body>
</html>
    """


def build_dashboard_fallback_html(payload: dict[str, Any]) -> str:
    title = escape(str(payload.get("period_label", "历史区间")))
    sample_count = int(payload.get("sample_count", 0) or 0)
    averages = payload.get("averages", {})
    top_emotion = escape(str(payload.get("top_emotion", "稳定")))
    top_signal = escape(str(payload.get("top_signal", "平稳")))
    trend_summary = escape(str(payload.get("trend_summary", "暂无趋势说明")))
    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <style>
    body {{ margin: 0; padding: 20px; background: #EAF7FF; color: #355E96; font-family: Inter, "Segoe UI", sans-serif; }}
    .card {{ background: rgba(226,242,255,0.78); border: 1px solid rgba(255,255,255,0.88); border-radius: 16px; padding: 16px; margin-bottom: 12px; }}
    .title {{ color: #2F5C98; font-size: 18px; font-weight: 600; margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
    .metric {{ background: rgba(206,232,255,0.64); border-radius: 14px; padding: 12px; }}
    .label {{ color: #6E8DB8; font-size: 12px; }}
    .value {{ color: #2F5C98; font-size: 28px; font-weight: 700; }}
    .sub {{ color: #6387B4; font-size: 12px; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="title">{title}</div>
    <div class="grid">
      <div class="metric"><div class="label">样本规模</div><div class="value">{sample_count}</div><div class="sub">历史记录</div></div>
      <div class="metric"><div class="label">平均压力</div><div class="value">{escape(str(averages.get('avg_stress', 0)))}</div><div class="sub">低饱和展示</div></div>
      <div class="metric"><div class="label">平均疲劳</div><div class="value">{escape(str(averages.get('avg_fatigue', 0)))}</div><div class="sub">低饱和展示</div></div>
    </div>
  </div>
  <div class="card">
    <div class="title">概览</div>
    <ul>
      <li>主情绪：{top_emotion}</li>
      <li>主信号：{top_signal}</li>
      <li>{trend_summary}</li>
    </ul>
  </div>
</body>
</html>
    """
