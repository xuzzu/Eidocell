/**
 * usePlotRenderer — Canvas/SVG hybrid rendering engine
 *
 * Base layer:  PixiJS WebGL canvas for high-performance data rendering (points, bars)
 * Overlay:     D3.js SVG for axes, labels, and interactive gate shapes
 */

import { ref, watch, onBeforeUnmount, type Ref, nextTick } from 'vue'
import { Application, Graphics, Container } from 'pixi.js'
import * as d3 from 'd3'
import type { ChartType, PlotData, GateOut } from '@/types'

// ── Types ───────────────────────────────────────────────────────────────

export interface PlotRendererOptions {
  container: Ref<HTMLElement | null>
  chartType: Ref<ChartType>
  plotData: Ref<PlotData | null>
  gates: Ref<GateOut[]>
  highlightedSampleIds?: Ref<Set<string> | null>
}

export interface GateDrawEvent {
  gateType: string
  definition: Record<string, unknown>
  parameters: string[]
}

export interface GateEditEvent {
  gateId: string
  definition: Record<string, unknown>
}

// Margins for axes
const MARGIN = { top: 12, right: 16, bottom: 40, left: 52 }

// ── Composable ──────────────────────────────────────────────────────────

export function usePlotRenderer(options: PlotRendererOptions) {
  const { container, chartType, plotData, gates, highlightedSampleIds } = options

  // Reactive state
  const activeTool = ref<string | null>(null)
  const isDrawing = ref(false)
  const drawPreviewColor = ref('#E53E3E')

  // Internal refs
  let pixi: Application | null = null
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null
  let dataContainer: Container | null = null
  let xScale: d3.ScaleLinear<number, number> = d3.scaleLinear()
  let yScale: d3.ScaleLinear<number, number> = d3.scaleLinear()
  let width = 0
  let height = 0
  let resizeObserver: ResizeObserver | null = null

  // Once domains are computed from the first non-empty data they stay frozen.
  // Why: backend filters out samples outside active gates, so re-deriving the
  // domain after each gate would shrink the visible region around the gate.
  let xDomainLocked = false
  let yDomainLocked = false

  // Gate drawing callbacks
  let onGateDrawn: ((e: GateDrawEvent) => void) | null = null
  let onGateEdited: ((e: GateEditEvent) => void) | null = null

  // Drawing state
  let drawStart: { x: number; y: number } | null = null
  let drawingShape: d3.Selection<SVGElement, unknown, null, undefined> | null = null
  let polygonVertices: Array<[number, number]> = []

  // ── Init ──────────────────────────────────────────────────────────────

  async function init() {
    const el = container.value
    if (!el) return

    const rect = el.getBoundingClientRect()
    width = rect.width
    height = rect.height
    if (width === 0 || height === 0) return

    // Create PixiJS application
    pixi = new Application()
    await pixi.init({
      width,
      height,
      backgroundAlpha: 0,
      antialias: true,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
    })

    // Style the canvas
    const canvas = pixi.canvas as HTMLCanvasElement
    canvas.style.position = 'absolute'
    canvas.style.top = '0'
    canvas.style.left = '0'
    canvas.style.width = '100%'
    canvas.style.height = '100%'
    el.appendChild(canvas)

    // Data container inside PixiJS
    dataContainer = new Container()
    pixi.stage.addChild(dataContainer)

    // Create SVG overlay via D3
    svg = d3.select(el)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('position', 'absolute')
      .style('top', '0')
      .style('left', '0')
      .style('pointer-events', 'all')

    // SVG groups for layers
    svg.append('g').attr('class', 'axes-layer')
    svg.append('g').attr('class', 'gates-layer')
    svg.append('g').attr('class', 'drawing-layer')

    // Setup scales
    updateScales()

    // Render
    renderData()
    renderAxes()
    renderGates()
    setupInteraction()

    // Resize observer
    resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect
        if (w > 0 && h > 0 && (w !== width || h !== height)) {
          width = w
          height = h
          onResize()
        }
      }
    })
    resizeObserver.observe(el)
  }

  function destroy() {
    resizeObserver?.disconnect()
    resizeObserver = null
    if (pixi) {
      pixi.destroy(true, { children: true })
      pixi = null
    }
    if (svg) {
      svg.remove()
      svg = null
    }
    dataContainer = null
    drawStart = null
    drawingShape = null
    polygonVertices = []
    xDomainLocked = false
    yDomainLocked = false
  }

  // ── Resize ────────────────────────────────────────────────────────────

  function onResize() {
    if (!pixi || !svg) return
    pixi.renderer.resize(width, height)
    svg.attr('width', width).attr('height', height)
    updateScales()
    renderData()
    renderAxes()
    renderGates()
  }

  // ── Scales ────────────────────────────────────────────────────────────

  function updateScales() {
    const data = plotData.value
    if (!data || data.data.length === 0) return

    const plotW = width - MARGIN.left - MARGIN.right
    const plotH = height - MARGIN.top - MARGIN.bottom
    const xRange: [number, number] = [MARGIN.left, MARGIN.left + plotW]
    const yRange: [number, number] = [MARGIN.top + plotH, MARGIN.top]

    // If domains are already locked, only refresh ranges (resize case).
    if (xDomainLocked) {
      xScale = xScale.copy().range(xRange)
    }
    if (yDomainLocked && chartType.value !== 'histogram') {
      yScale = yScale.copy().range(yRange)
    }
    if (xDomainLocked && (yDomainLocked || chartType.value === 'histogram')) {
      // Histogram y-domain is recomputed per-render in renderHistogram, so we
      // only need to ensure its range is current.
      if (chartType.value === 'histogram') {
        yScale = yScale.copy().range(yRange)
      }
      return
    }

    const xVar = data.parameters.x_variable as string
    const yVar = data.parameters.y_variable as string | undefined

    if (!xDomainLocked) {
      const xVals = data.data.map(p => p.values[xVar]).filter(v => v != null)
      if (xVals.length === 0) return
      const xMin = d3.min(xVals)!
      const xMax = d3.max(xVals)!
      const xPad = (xMax - xMin) * 0.02 || 1
      xScale = d3.scaleLinear()
        .domain([xMin - xPad, xMax + xPad])
        .range(xRange)
      xDomainLocked = true
    }

    if (chartType.value === 'histogram') {
      yScale = d3.scaleLinear().range(yRange)
    } else if (!yDomainLocked) {
      const yVals = data.data.map(p => p.values[yVar!]).filter(v => v != null)
      if (yVals.length === 0) return
      const yMin = d3.min(yVals)!
      const yMax = d3.max(yVals)!
      const yPad = (yMax - yMin) * 0.02 || 1
      yScale = d3.scaleLinear()
        .domain([yMin - yPad, yMax + yPad])
        .range(yRange)
      yDomainLocked = true
    }
  }

  // ── Data Rendering (PixiJS) ───────────────────────────────────────────

  function renderData() {
    if (!pixi || !dataContainer) return
    dataContainer.removeChildren()
    // Clean up previous contour SVG layer
    svg?.selectAll('.contour-layer').remove()

    const data = plotData.value
    if (!data || data.data.length === 0) return

    if (chartType.value === 'histogram') {
      renderHistogram(data)
    } else if (chartType.value === 'density') {
      renderDensity(data)
    } else if (chartType.value === 'contour') {
      renderScatter(data) // render points faintly as background
      renderContourOverlay(data)
    } else {
      renderScatter(data)
    }
  }

  function renderHistogram(data: PlotData) {
    if (!dataContainer) return
    const xVar = data.parameters.x_variable as string
    const values = data.data.map(p => p.values[xVar]).filter(v => v != null)
    if (values.length === 0) return

    const numBins = (data.parameters.num_bins as number) ?? 30

    const binner = d3.bin()
      .domain(xScale.domain() as [number, number])
      .thresholds(numBins)

    const bins = binner(values)
    const maxCount = d3.max(bins, b => b.length) || 1

    yScale.domain([0, maxCount * 1.05])

    const highlighted = highlightedSampleIds?.value
    const sampleIdsByValue = new Map<number, string>()
    if (highlighted) {
      data.data.forEach(p => {
        const v = p.values[xVar]
        if (v != null) sampleIdsByValue.set(v, p.sample_id)
      })
    }

    for (const bin of bins) {
      if (bin.length === 0) continue
      const x0 = xScale(bin.x0!)
      const x1 = xScale(bin.x1!)
      const y0 = yScale(0)
      const y1 = yScale(bin.length)
      const barW = Math.max(1, x1 - x0 - 1)
      const barH = Math.max(0, y0 - y1)

      const bar = new Graphics()
      bar.rect(x0, y1, barW, barH)
      bar.fill({ color: 0x404040, alpha: 0.7 })
      bar.stroke({ color: 0x303030, width: 0.5, alpha: 0.5 })
      dataContainer!.addChild(bar)
    }
  }

  function renderScatter(data: PlotData) {
    if (!dataContainer) return
    const xVar = data.parameters.x_variable as string
    const yVar = data.parameters.y_variable as string

    const highlighted = highlightedSampleIds?.value
    const pointSize = data.data.length > 10000 ? 1.5 : data.data.length > 1000 ? 2 : 3

    // Batch points in a single Graphics for performance
    const gNormal = new Graphics()
    const gHighlight = new Graphics()

    for (const pt of data.data) {
      const xVal = pt.values[xVar]
      const yVal = pt.values[yVar]
      if (xVal == null || yVal == null) continue

      const px = xScale(xVal)
      const py = yScale(yVal)

      // Determine color
      let color = 0x64748b // default slate
      if (pt.class_color) {
        color = parseInt(pt.class_color.replace('#', ''), 16)
      }

      if (highlighted && highlighted.size > 0) {
        if (highlighted.has(pt.sample_id)) {
          gHighlight.circle(px, py, pointSize * 1.3)
          gHighlight.fill({ color, alpha: 0.85 })
        } else {
          gNormal.circle(px, py, pointSize)
          gNormal.fill({ color, alpha: 0.08 })
        }
      } else {
        gNormal.circle(px, py, pointSize)
        gNormal.fill({ color, alpha: 0.55 })
      }
    }

    dataContainer!.addChild(gNormal)
    dataContainer!.addChild(gHighlight)
  }

  // ── Density Heatmap (PixiJS) ───────────────────────────────────────────

  function renderDensity(data: PlotData) {
    if (!dataContainer) return
    const xVar = data.parameters.x_variable as string
    const yVar = data.parameters.y_variable as string
    if (!yVar) return

    const plotW = width - MARGIN.left - MARGIN.right
    const plotH = height - MARGIN.top - MARGIN.bottom

    // Number of bins in each axis
    const nBinsX = Math.max(8, Math.min(80, Math.round(plotW / 6)))
    const nBinsY = Math.max(8, Math.min(80, Math.round(plotH / 6)))

    // Build 2D bin grid
    const grid = new Float64Array(nBinsX * nBinsY)
    const xDomain = xScale.domain()
    const yDomain = yScale.domain()
    const xStep = (xDomain[1] - xDomain[0]) / nBinsX
    const yStep = (yDomain[1] - yDomain[0]) / nBinsY

    for (const pt of data.data) {
      const xVal = pt.values[xVar]
      const yVal = pt.values[yVar]
      if (xVal == null || yVal == null) continue
      const xi = Math.floor((xVal - xDomain[0]) / xStep)
      const yi = Math.floor((yVal - yDomain[0]) / yStep)
      if (xi >= 0 && xi < nBinsX && yi >= 0 && yi < nBinsY) {
        grid[yi * nBinsX + xi]++
      }
    }

    const maxCount = Math.max(1, ...grid)

    // Viridis-inspired color scale via D3
    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, maxCount])

    const cellW = plotW / nBinsX
    const cellH = plotH / nBinsY
    const g = new Graphics()

    for (let yi = 0; yi < nBinsY; yi++) {
      for (let xi = 0; xi < nBinsX; xi++) {
        const count = grid[yi * nBinsX + xi]
        if (count === 0) continue
        const color = colorScale(count)
        const rgb = d3.rgb(color)
        const hex = (rgb.r << 16) | (rgb.g << 8) | rgb.b

        const px = MARGIN.left + xi * cellW
        // Y is inverted (screen coords)
        const py = MARGIN.top + plotH - (yi + 1) * cellH

        g.rect(px, py, cellW + 0.5, cellH + 0.5)
        g.fill({ color: hex, alpha: 0.85 })
      }
    }

    dataContainer!.addChild(g)
  }

  // ── Contour Overlay (D3 SVG) ─────────────────────────────────────────

  function renderContourOverlay(data: PlotData) {
    if (!svg) return
    const xVar = data.parameters.x_variable as string
    const yVar = data.parameters.y_variable as string
    if (!yVar) return

    // Remove previous contours
    svg.selectAll('.contour-layer').remove()
    const contourLayer = svg.insert('g', '.gates-layer').attr('class', 'contour-layer')

    const plotW = width - MARGIN.left - MARGIN.right
    const plotH = height - MARGIN.top - MARGIN.bottom
    const gridW = Math.min(80, Math.round(plotW / 4))
    const gridH = Math.min(80, Math.round(plotH / 4))

    // Build density grid in pixel-space
    const grid = new Float64Array(gridW * gridH)
    const bandwidth = 10 // Gaussian blur radius in grid cells

    for (const pt of data.data) {
      const xVal = pt.values[xVar]
      const yVal = pt.values[yVar]
      if (xVal == null || yVal == null) continue
      const px = (xScale(xVal) - MARGIN.left) / plotW * gridW
      const py = (yScale(yVal) - MARGIN.top) / plotH * gridH
      const gi = Math.floor(px)
      const gj = Math.floor(py)
      if (gi >= 0 && gi < gridW && gj >= 0 && gj < gridH) {
        grid[gj * gridW + gi]++
      }
    }

    // Simple box blur for smoothing (2 passes)
    const blurred = boxBlur2D(grid, gridW, gridH, Math.max(1, Math.round(bandwidth / 3)))
    const blurred2 = boxBlur2D(blurred, gridW, gridH, Math.max(1, Math.round(bandwidth / 3)))

    const maxVal = Math.max(1, ...blurred2)
    const thresholds = d3.range(0.05, 1.0, 0.1).map(t => t * maxVal)

    const contourGenerator = d3.contours()
      .size([gridW, gridH])
      .thresholds(thresholds)

    const contours = contourGenerator(Array.from(blurred2))

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, maxVal])

    // Scale contour coordinates from grid space to pixel space
    const scaleX = plotW / gridW
    const scaleY = plotH / gridH

    for (const contour of contours) {
      const color = colorScale(contour.value)
      contourLayer.append('path')
        .attr('d', d3.geoPath(
          d3.geoTransform({
            point(x, y) {
              this.stream.point(MARGIN.left + x * scaleX, MARGIN.top + y * scaleY)
            },
          }),
        )(contour))
        .attr('fill', color)
        .attr('fill-opacity', 0.15)
        .attr('stroke', color)
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', 1)
    }
  }

  function boxBlur2D(input: Float64Array, w: number, h: number, r: number): Float64Array {
    const out = new Float64Array(w * h)
    // Horizontal pass
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0
        let count = 0
        for (let dx = -r; dx <= r; dx++) {
          const nx = x + dx
          if (nx >= 0 && nx < w) { sum += input[y * w + nx]; count++ }
        }
        out[y * w + x] = sum / count
      }
    }
    // Vertical pass
    const out2 = new Float64Array(w * h)
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let sum = 0
        let count = 0
        for (let dy = -r; dy <= r; dy++) {
          const ny = y + dy
          if (ny >= 0 && ny < h) { sum += out[ny * w + x]; count++ }
        }
        out2[y * w + x] = sum / count
      }
    }
    return out2
  }

  // ── Axes (D3 SVG) ────────────────────────────────────────────────────

  function renderAxes() {
    if (!svg) return
    const axesLayer = svg.select<SVGGElement>('.axes-layer')
    axesLayer.selectAll('*').remove()

    const data = plotData.value
    if (!data) return

    const plotH = height - MARGIN.top - MARGIN.bottom
    const xVar = data.parameters.x_variable as string
    const yVar = data.parameters.y_variable as string | undefined

    // X axis
    const xAxis = d3.axisBottom(xScale).ticks(Math.max(3, Math.floor((width - MARGIN.left - MARGIN.right) / 80)))
    axesLayer.append('g')
      .attr('transform', `translate(0, ${MARGIN.top + plotH})`)
      .call(xAxis)
      .call(g => {
        g.selectAll('text')
          .attr('font-size', '9px')
          .attr('font-family', 'ui-monospace, monospace')
          .attr('fill', 'currentColor')
          .attr('opacity', 0.5)
        g.selectAll('line').attr('stroke', 'currentColor').attr('opacity', 0.15)
        g.select('.domain').attr('stroke', 'currentColor').attr('opacity', 0.2)
      })

    // X axis label
    axesLayer.append('text')
      .attr('x', MARGIN.left + (width - MARGIN.left - MARGIN.right) / 2)
      .attr('y', height - 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .attr('font-family', 'ui-monospace, monospace')
      .attr('font-weight', '600')
      .attr('letter-spacing', '0.05em')
      .attr('fill', 'currentColor')
      .attr('opacity', 0.45)
      .text(xVar.toUpperCase().replace(/_/g, ' '))

    // Y axis
    const yAxis = d3.axisLeft(yScale).ticks(Math.max(3, Math.floor(plotH / 50)))
    axesLayer.append('g')
      .attr('transform', `translate(${MARGIN.left}, 0)`)
      .call(yAxis)
      .call(g => {
        g.selectAll('text')
          .attr('font-size', '9px')
          .attr('font-family', 'ui-monospace, monospace')
          .attr('fill', 'currentColor')
          .attr('opacity', 0.5)
        g.selectAll('line').attr('stroke', 'currentColor').attr('opacity', 0.15)
        g.select('.domain').attr('stroke', 'currentColor').attr('opacity', 0.2)
      })

    // Y axis label
    if (chartType.value === 'histogram') {
      axesLayer.append('text')
        .attr('transform', `rotate(-90)`)
        .attr('x', -(MARGIN.top + plotH / 2))
        .attr('y', 12)
        .attr('text-anchor', 'middle')
        .attr('font-size', '9px')
        .attr('font-family', 'ui-monospace, monospace')
        .attr('font-weight', '600')
        .attr('letter-spacing', '0.05em')
        .attr('fill', 'currentColor')
        .attr('opacity', 0.45)
        .text('COUNT')
    } else if (yVar) {
      axesLayer.append('text')
        .attr('transform', `rotate(-90)`)
        .attr('x', -(MARGIN.top + plotH / 2))
        .attr('y', 12)
        .attr('text-anchor', 'middle')
        .attr('font-size', '9px')
        .attr('font-family', 'ui-monospace, monospace')
        .attr('font-weight', '600')
        .attr('letter-spacing', '0.05em')
        .attr('fill', 'currentColor')
        .attr('opacity', 0.45)
        .text(yVar.toUpperCase().replace(/_/g, ' '))
    }
  }

  // ── Gate Rendering (D3 SVG) ───────────────────────────────────────────

  function renderGates() {
    if (!svg) return
    const gatesLayer = svg.select<SVGGElement>('.gates-layer')
    gatesLayer.selectAll('*').remove()

    for (const gate of gates.value) {
      renderSingleGate(gatesLayer, gate)
    }
  }

  function renderSingleGate(
    layer: d3.Selection<SVGGElement, unknown, null, undefined>,
    gate: GateOut,
  ) {
    const g = layer.append('g').attr('class', `gate gate-${gate.id}`)
    const color = gate.color || '#FF0000'

    if (gate.gate_type === 'rectangular') {
      const { x, y: gy, width: gw, height: gh } = gate.definition as {
        x: number; y: number; width: number; height: number
      }
      const px = xScale(x)
      const py = yScale(gy + gh)
      const pw = xScale(x + gw) - px
      const ph = yScale(gy) - py

      const rect = g.append('rect')
        .attr('x', px).attr('y', py)
        .attr('width', Math.abs(pw)).attr('height', Math.abs(ph))
        .attr('fill', color).attr('fill-opacity', 0.12)
        .attr('stroke', color).attr('stroke-opacity', 0.7)
        .attr('stroke-width', 1.5)
        .attr('cursor', 'move')

      // Make draggable
      setupRectDrag(rect, gate)

      // Resize handles
      const handles = [
        { cx: px, cy: py, cursor: 'nw-resize', anchor: 'nw' },
        { cx: px + pw, cy: py, cursor: 'ne-resize', anchor: 'ne' },
        { cx: px, cy: py + ph, cursor: 'sw-resize', anchor: 'sw' },
        { cx: px + pw, cy: py + ph, cursor: 'se-resize', anchor: 'se' },
      ]
      for (const h of handles) {
        g.append('rect')
          .attr('x', h.cx - 3).attr('y', h.cy - 3)
          .attr('width', 6).attr('height', 6)
          .attr('fill', color).attr('fill-opacity', 0.9)
          .attr('stroke', '#fff').attr('stroke-width', 0.5)
          .attr('cursor', h.cursor)
          .call(setupResizeHandleDrag(gate, h.anchor))
      }
    } else if (gate.gate_type === 'interval') {
      const { min, max } = gate.definition as { min: number; max: number }
      const px0 = xScale(min)
      const px1 = xScale(max)
      const plotTop = MARGIN.top
      const plotBottom = height - MARGIN.bottom

      g.append('rect')
        .attr('x', px0).attr('y', plotTop)
        .attr('width', px1 - px0).attr('height', plotBottom - plotTop)
        .attr('fill', color).attr('fill-opacity', 0.12)
        .attr('stroke', 'none')

      // Left edge handle
      g.append('line')
        .attr('x1', px0).attr('x2', px0)
        .attr('y1', plotTop).attr('y2', plotBottom)
        .attr('stroke', color).attr('stroke-width', 2)
        .attr('stroke-opacity', 0.7)
        .attr('cursor', 'ew-resize')
        .call(setupIntervalEdgeDrag(gate, 'min'))

      // Right edge handle
      g.append('line')
        .attr('x1', px1).attr('x2', px1)
        .attr('y1', plotTop).attr('y2', plotBottom)
        .attr('stroke', color).attr('stroke-width', 2)
        .attr('stroke-opacity', 0.7)
        .attr('cursor', 'ew-resize')
        .call(setupIntervalEdgeDrag(gate, 'max'))
    } else if (gate.gate_type === 'polygon') {
      const vertices = gate.definition.vertices as Array<[number, number]>
      const points = vertices.map(([vx, vy]) => `${xScale(vx)},${yScale(vy)}`).join(' ')

      g.append('polygon')
        .attr('points', points)
        .attr('fill', color).attr('fill-opacity', 0.12)
        .attr('stroke', color).attr('stroke-opacity', 0.7)
        .attr('stroke-width', 1.5)

      // Vertex handles
      for (let i = 0; i < vertices.length; i++) {
        const [vx, vy] = vertices[i]
        g.append('circle')
          .attr('cx', xScale(vx)).attr('cy', yScale(vy))
          .attr('r', 4)
          .attr('fill', color).attr('fill-opacity', 0.9)
          .attr('stroke', '#fff').attr('stroke-width', 0.5)
          .attr('cursor', 'pointer')
          .call(setupPolygonVertexDrag(gate, i))
      }
    } else if (gate.gate_type === 'ellipse') {
      const { cx, cy, rx, ry } = gate.definition as {
        cx: number; cy: number; rx: number; ry: number
      }
      const pcx = xScale(cx)
      const pcy = yScale(cy)
      const prx = Math.abs(xScale(cx + rx) - pcx)
      const pry = Math.abs(yScale(cy + ry) - pcy)

      const ellipse = g.append('ellipse')
        .attr('cx', pcx).attr('cy', pcy)
        .attr('rx', prx).attr('ry', pry)
        .attr('fill', color).attr('fill-opacity', 0.12)
        .attr('stroke', color).attr('stroke-opacity', 0.7)
        .attr('stroke-width', 1.5)
        .attr('cursor', 'move')

      setupEllipseDrag(ellipse, gate)

      // Radius handles: right (rx) and top (ry)
      g.append('rect')
        .attr('x', pcx + prx - 3).attr('y', pcy - 3)
        .attr('width', 6).attr('height', 6)
        .attr('fill', color).attr('fill-opacity', 0.9)
        .attr('stroke', '#fff').attr('stroke-width', 0.5)
        .attr('cursor', 'ew-resize')
        .call(setupEllipseRadiusDrag(gate, 'rx'))

      g.append('rect')
        .attr('x', pcx - 3).attr('y', pcy - pry - 3)
        .attr('width', 6).attr('height', 6)
        .attr('fill', color).attr('fill-opacity', 0.9)
        .attr('stroke', '#fff').attr('stroke-width', 0.5)
        .attr('cursor', 'ns-resize')
        .call(setupEllipseRadiusDrag(gate, 'ry'))
    } else if (gate.gate_type === 'quadrant') {
      const { x_threshold, y_threshold } = gate.definition as {
        x_threshold: number; y_threshold: number
      }
      const px = xScale(x_threshold)
      const py = yScale(y_threshold)
      const plotTop = MARGIN.top
      const plotBottom = height - MARGIN.bottom
      const plotLeft = MARGIN.left
      const plotRight = width - MARGIN.right

      g.append('line')
        .attr('x1', px).attr('x2', px)
        .attr('y1', plotTop).attr('y2', plotBottom)
        .attr('stroke', color).attr('stroke-width', 1.5).attr('stroke-opacity', 0.7)
        .attr('stroke-dasharray', '4,3')

      g.append('line')
        .attr('x1', plotLeft).attr('x2', plotRight)
        .attr('y1', py).attr('y2', py)
        .attr('stroke', color).attr('stroke-width', 1.5).attr('stroke-opacity', 0.7)
        .attr('stroke-dasharray', '4,3')
    }

    // Gate label
    const labelPos = getGateLabelPosition(gate)
    if (labelPos) {
      g.append('text')
        .attr('x', labelPos.x).attr('y', labelPos.y)
        .attr('font-size', '9px')
        .attr('font-family', 'ui-monospace, monospace')
        .attr('font-weight', '700')
        .attr('fill', color)
        .attr('opacity', 0.85)
        .text(`${gate.name} (${gate.percentage.toFixed(1)}%)`)
    }
  }

  function getGateLabelPosition(gate: GateOut): { x: number; y: number } | null {
    if (gate.gate_type === 'rectangular') {
      const d = gate.definition as { x: number; y: number; width: number; height: number }
      return { x: xScale(d.x) + 4, y: yScale(d.y + d.height) - 4 }
    }
    if (gate.gate_type === 'interval') {
      const d = gate.definition as { min: number; max: number }
      return { x: xScale(d.min) + 4, y: MARGIN.top + 12 }
    }
    if (gate.gate_type === 'polygon') {
      const v = gate.definition.vertices as Array<[number, number]>
      const minX = Math.min(...v.map(p => p[0]))
      const maxY = Math.max(...v.map(p => p[1]))
      return { x: xScale(minX) + 4, y: yScale(maxY) - 4 }
    }
    if (gate.gate_type === 'ellipse') {
      const d = gate.definition as { cx: number; cy: number; rx: number; ry: number }
      return { x: xScale(d.cx - d.rx) + 4, y: yScale(d.cy + d.ry) - 4 }
    }
    if (gate.gate_type === 'quadrant') {
      const d = gate.definition as { x_threshold: number; y_threshold: number }
      return { x: xScale(d.x_threshold) + 4, y: yScale(d.y_threshold) - 12 }
    }
    return null
  }

  // ── Drag Handlers for Gate Editing ────────────────────────────────────

  function setupRectDrag(
    selection: d3.Selection<SVGRectElement, unknown, null, undefined>,
    gate: GateOut,
  ) {
    const drag = d3.drag<SVGRectElement, unknown>()
      .on('start', function () {
        d3.select(this).attr('stroke-opacity', 1)
      })
      .on('drag', function (event) {
        const el = d3.select(this)
        const newX = parseFloat(el.attr('x')) + event.dx
        const newY = parseFloat(el.attr('y')) + event.dy
        el.attr('x', newX).attr('y', newY)
      })
      .on('end', function () {
        d3.select(this).attr('stroke-opacity', 0.7)
        const el = d3.select(this)
        const x = xScale.invert(parseFloat(el.attr('x')))
        const w = xScale.invert(parseFloat(el.attr('x')) + parseFloat(el.attr('width'))) - x
        const yBottom = yScale.invert(parseFloat(el.attr('y')) + parseFloat(el.attr('height')))
        const h = yScale.invert(parseFloat(el.attr('y'))) - yBottom
        emitGateEdit(gate.id, { x, y: yBottom, width: w, height: h })
      })
    selection.call(drag)
  }

  function setupResizeHandleDrag(gate: GateOut, anchor: string) {
    return function (selection: d3.Selection<SVGRectElement, unknown, null, undefined>) {
      const drag = d3.drag<SVGRectElement, unknown>()
        .on('drag', function (event) {
          // Find the parent gate rect
          const gateGroup = d3.select(this.parentNode as SVGGElement)
          const rect = gateGroup.select<SVGRectElement>('rect:first-child')
          let x = parseFloat(rect.attr('x'))
          let y = parseFloat(rect.attr('y'))
          let w = parseFloat(rect.attr('width'))
          let h = parseFloat(rect.attr('height'))

          if (anchor.includes('w')) { x += event.dx; w -= event.dx }
          if (anchor.includes('e')) { w += event.dx }
          if (anchor.includes('n')) { y += event.dy; h -= event.dy }
          if (anchor.includes('s')) { h += event.dy }

          if (w > 4 && h > 4) {
            rect.attr('x', x).attr('y', y).attr('width', w).attr('height', h)
          }
        })
        .on('end', function () {
          const gateGroup = d3.select(this.parentNode as SVGGElement)
          const rect = gateGroup.select<SVGRectElement>('rect:first-child')
          const px = parseFloat(rect.attr('x'))
          const py = parseFloat(rect.attr('y'))
          const pw = parseFloat(rect.attr('width'))
          const ph = parseFloat(rect.attr('height'))
          const dataX = xScale.invert(px)
          const dataW = xScale.invert(px + pw) - dataX
          const dataYBottom = yScale.invert(py + ph)
          const dataH = yScale.invert(py) - dataYBottom
          emitGateEdit(gate.id, { x: dataX, y: dataYBottom, width: dataW, height: dataH })
        })
      selection.call(drag)
    }
  }

  function setupIntervalEdgeDrag(gate: GateOut, edge: 'min' | 'max') {
    return function (selection: d3.Selection<SVGLineElement, unknown, null, undefined>) {
      const drag = d3.drag<SVGLineElement, unknown>()
        .on('drag', function (event) {
          const newX = parseFloat(d3.select(this).attr('x1')) + event.dx
          d3.select(this).attr('x1', newX).attr('x2', newX)
        })
        .on('end', function () {
          const px = parseFloat(d3.select(this).attr('x1'))
          const val = xScale.invert(px)
          const defn = { ...gate.definition } as { min: number; max: number }
          defn[edge] = val
          if (defn.min > defn.max) [defn.min, defn.max] = [defn.max, defn.min]
          emitGateEdit(gate.id, defn)
        })
      selection.call(drag)
    }
  }

  function setupEllipseDrag(
    selection: d3.Selection<SVGEllipseElement, unknown, null, undefined>,
    gate: GateOut,
  ) {
    const drag = d3.drag<SVGEllipseElement, unknown>()
      .on('start', function () {
        d3.select(this).attr('stroke-opacity', 1)
      })
      .on('drag', function (event) {
        const el = d3.select(this)
        const newCx = parseFloat(el.attr('cx')) + event.dx
        const newCy = parseFloat(el.attr('cy')) + event.dy
        el.attr('cx', newCx).attr('cy', newCy)
      })
      .on('end', function () {
        d3.select(this).attr('stroke-opacity', 0.7)
        const el = d3.select(this)
        const dataCx = xScale.invert(parseFloat(el.attr('cx')))
        const dataCy = yScale.invert(parseFloat(el.attr('cy')))
        const def = gate.definition as { rx: number; ry: number; angle?: number }
        emitGateEdit(gate.id, {
          cx: dataCx, cy: dataCy,
          rx: def.rx, ry: def.ry, angle: def.angle ?? 0,
        })
      })
    selection.call(drag)
  }

  function setupEllipseRadiusDrag(gate: GateOut, axis: 'rx' | 'ry') {
    return function (selection: d3.Selection<SVGRectElement, unknown, null, undefined>) {
      const drag = d3.drag<SVGRectElement, unknown>()
        .on('drag', function (event) {
          const ellipseEl = d3.select(this.parentNode as SVGGElement).select<SVGEllipseElement>('ellipse')
          if (axis === 'rx') {
            const newRx = Math.max(2, parseFloat(ellipseEl.attr('rx')) + event.dx)
            ellipseEl.attr('rx', newRx)
            const handle = d3.select(this)
            handle.attr('x', parseFloat(ellipseEl.attr('cx')) + newRx - 3)
          } else {
            const newRy = Math.max(2, parseFloat(ellipseEl.attr('ry')) - event.dy)
            ellipseEl.attr('ry', newRy)
            const handle = d3.select(this)
            handle.attr('y', parseFloat(ellipseEl.attr('cy')) - newRy - 3)
          }
        })
        .on('end', function () {
          const ellipseEl = d3.select(this.parentNode as SVGGElement).select<SVGEllipseElement>('ellipse')
          const pcx = parseFloat(ellipseEl.attr('cx'))
          const pcy = parseFloat(ellipseEl.attr('cy'))
          const prx = parseFloat(ellipseEl.attr('rx'))
          const pry = parseFloat(ellipseEl.attr('ry'))
          const dataCx = xScale.invert(pcx)
          const dataCy = yScale.invert(pcy)
          const dataRx = Math.abs(xScale.invert(pcx + prx) - dataCx)
          const dataRy = Math.abs(yScale.invert(pcy - pry) - dataCy)
          const def = gate.definition as { angle?: number }
          emitGateEdit(gate.id, {
            cx: dataCx, cy: dataCy,
            rx: dataRx, ry: dataRy, angle: def.angle ?? 0,
          })
        })
      selection.call(drag)
    }
  }

  function setupPolygonVertexDrag(gate: GateOut, vertexIndex: number) {
    return function (selection: d3.Selection<SVGCircleElement, unknown, null, undefined>) {
      const drag = d3.drag<SVGCircleElement, unknown>()
        .on('drag', function (event) {
          const newCx = parseFloat(d3.select(this).attr('cx')) + event.dx
          const newCy = parseFloat(d3.select(this).attr('cy')) + event.dy
          d3.select(this).attr('cx', newCx).attr('cy', newCy)
        })
        .on('end', function () {
          const cx = parseFloat(d3.select(this).attr('cx'))
          const cy = parseFloat(d3.select(this).attr('cy'))
          const vertices = [...(gate.definition.vertices as Array<[number, number]>)]
          vertices[vertexIndex] = [xScale.invert(cx), yScale.invert(cy)]
          emitGateEdit(gate.id, { vertices })
        })
      selection.call(drag)
    }
  }

  // ── Gate Drawing Interaction ──────────────────────────────────────────

  function setupInteraction() {
    if (!svg) return
    const drawLayer = svg.select<SVGGElement>('.drawing-layer')

    svg
      .on('mousedown.draw', function (event: MouseEvent) {
        if (!activeTool.value || event.button !== 0) return
        // Only respond within plot area
        const [mx, my] = d3.pointer(event)
        if (mx < MARGIN.left || mx > width - MARGIN.right ||
            my < MARGIN.top || my > height - MARGIN.bottom) return

        const tool = activeTool.value

        if (tool === 'polygon') {
          // Polygon: accumulate vertices on click
          return
        }

        isDrawing.value = true
        drawStart = { x: mx, y: my }

        const pc = drawPreviewColor.value

        if (tool === 'rectangular') {
          drawingShape = drawLayer.append('rect')
            .attr('x', mx).attr('y', my)
            .attr('width', 0).attr('height', 0)
            .attr('fill', pc).attr('fill-opacity', 0.1)
            .attr('stroke', pc)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '4,3') as any
        } else if (tool === 'interval') {
          drawingShape = drawLayer.append('rect')
            .attr('x', mx).attr('y', MARGIN.top)
            .attr('width', 0).attr('height', height - MARGIN.top - MARGIN.bottom)
            .attr('fill', pc).attr('fill-opacity', 0.1)
            .attr('stroke', pc)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '4,3') as any
        } else if (tool === 'ellipse') {
          drawingShape = drawLayer.append('ellipse')
            .attr('cx', mx).attr('cy', my)
            .attr('rx', 0).attr('ry', 0)
            .attr('fill', pc).attr('fill-opacity', 0.1)
            .attr('stroke', pc)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '4,3') as any
        }
      })
      .on('mousemove.draw', function (event: MouseEvent) {
        if (!isDrawing.value || !drawStart || !drawingShape) return
        const [mx, my] = d3.pointer(event)
        const tool = activeTool.value

        if (tool === 'rectangular') {
          const x = Math.min(drawStart.x, mx)
          const y = Math.min(drawStart.y, my)
          const w = Math.abs(mx - drawStart.x)
          const h = Math.abs(my - drawStart.y)
          drawingShape.attr('x', x).attr('y', y).attr('width', w).attr('height', h)
        } else if (tool === 'interval') {
          const x = Math.min(drawStart.x, mx)
          const w = Math.abs(mx - drawStart.x)
          drawingShape.attr('x', x).attr('width', w)
        } else if (tool === 'ellipse') {
          const rx = Math.abs(mx - drawStart.x)
          const ry = Math.abs(my - drawStart.y)
          drawingShape.attr('rx', rx).attr('ry', ry)
        }
      })
      .on('mouseup.draw', function (event: MouseEvent) {
        if (!isDrawing.value || !drawStart) return
        const [mx, my] = d3.pointer(event)
        const tool = activeTool.value

        // Clean up drawing shape
        drawingShape?.remove()
        drawingShape = null
        isDrawing.value = false

        const data = plotData.value
        if (!data) return

        const xVar = data.parameters.x_variable as string
        const yVar = data.parameters.y_variable as string

        if (tool === 'rectangular' && Math.abs(mx - drawStart.x) > 5 && Math.abs(my - drawStart.y) > 5) {
          const x0 = xScale.invert(Math.min(drawStart.x, mx))
          const x1 = xScale.invert(Math.max(drawStart.x, mx))
          const y0 = yScale.invert(Math.max(drawStart.y, my))
          const y1 = yScale.invert(Math.min(drawStart.y, my))
          emitGateDraw({
            gateType: 'rectangular',
            definition: { x: x0, y: y0, width: x1 - x0, height: y1 - y0 },
            parameters: [xVar, yVar],
          })
        } else if (tool === 'interval' && Math.abs(mx - drawStart.x) > 5) {
          const min = xScale.invert(Math.min(drawStart.x, mx))
          const max = xScale.invert(Math.max(drawStart.x, mx))
          emitGateDraw({
            gateType: 'interval',
            definition: { min, max },
            parameters: [xVar],
          })
        } else if (tool === 'ellipse' && Math.abs(mx - drawStart.x) > 5 && Math.abs(my - drawStart.y) > 5) {
          const cx = xScale.invert(drawStart.x)
          const cy = yScale.invert(drawStart.y)
          const rx = Math.abs(xScale.invert(mx) - cx)
          const ry = Math.abs(yScale.invert(my) - cy)
          emitGateDraw({
            gateType: 'ellipse',
            definition: { cx, cy, rx, ry, angle: 0 },
            parameters: [xVar, yVar],
          })
        }

        drawStart = null
      })

    // Polygon: click to add a vertex; click near the first vertex (when 3+
    // verts already exist) OR double-click anywhere to close. The first-vertex
    // close gives users an obvious target instead of needing dblclick precision.
    const FIRST_VERTEX_CLOSE_RADIUS = 10
    svg
      .on('click.polygon', function (event: MouseEvent) {
        if (activeTool.value !== 'polygon') return
        const [mx, my] = d3.pointer(event)
        if (mx < MARGIN.left || mx > width - MARGIN.right ||
            my < MARGIN.top || my > height - MARGIN.bottom) return

        if (polygonVertices.length >= 3) {
          const [fx, fy] = polygonVertices[0]
          const dx = mx - fx
          const dy = my - fy
          if (dx * dx + dy * dy <= FIRST_VERTEX_CLOSE_RADIUS * FIRST_VERTEX_CLOSE_RADIUS) {
            finalizePolygon(drawLayer)
            return
          }
        }

        polygonVertices.push([mx, my])
        redrawPolygonPreview(drawLayer)
      })
      .on('dblclick.polygon', function (event: MouseEvent) {
        if (activeTool.value !== 'polygon' || polygonVertices.length < 3) return
        event.preventDefault()
        event.stopPropagation()
        finalizePolygon(drawLayer)
      })

    // Quadrant: single click sets crosshair
    svg
      .on('click.quadrant', function (event: MouseEvent) {
        if (activeTool.value !== 'quadrant') return
        const [mx, my] = d3.pointer(event)
        if (mx < MARGIN.left || mx > width - MARGIN.right ||
            my < MARGIN.top || my > height - MARGIN.bottom) return

        const data = plotData.value
        if (!data) return

        const xVar = data.parameters.x_variable as string
        const yVar = data.parameters.y_variable as string
        const xThreshold = xScale.invert(mx)
        const yThreshold = yScale.invert(my)

        emitGateDraw({
          gateType: 'quadrant',
          definition: { x_threshold: xThreshold, y_threshold: yThreshold },
          parameters: [xVar, yVar],
        })
      })
  }

  function redrawPolygonPreview(
    layer: d3.Selection<SVGGElement, unknown, null, undefined>,
  ) {
    layer.selectAll('*').remove()
    if (polygonVertices.length === 0) return

    const pc = drawPreviewColor.value
    const closeable = polygonVertices.length >= 3

    if (polygonVertices.length > 1) {
      const lineData = polygonVertices.map(([x, y]) => `${x},${y}`).join(' ')
      layer.append('polyline')
        .attr('points', lineData)
        .attr('fill', 'none')
        .attr('stroke', pc)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3')
    }

    polygonVertices.forEach(([vx, vy], i) => {
      const isFirst = i === 0
      const r = isFirst && closeable ? 6 : 4
      layer.append('circle')
        .attr('cx', vx).attr('cy', vy)
        .attr('r', r)
        .attr('fill', pc).attr('fill-opacity', isFirst && closeable ? 1 : 0.8)
        .attr('stroke', '#fff').attr('stroke-width', isFirst && closeable ? 1.5 : 0.5)
    })
  }

  function finalizePolygon(
    layer: d3.Selection<SVGGElement, unknown, null, undefined>,
  ) {
    if (polygonVertices.length < 3) return
    const data = plotData.value
    if (!data) return
    const xVar = data.parameters.x_variable as string
    const yVar = data.parameters.y_variable as string
    const dataVerts = polygonVertices.map(([px, py]) => [
      xScale.invert(px), yScale.invert(py),
    ])
    emitGateDraw({
      gateType: 'polygon',
      definition: { vertices: dataVerts },
      parameters: [xVar, yVar],
    })
    polygonVertices = []
    layer.selectAll('*').remove()
  }

  // ── Event Emitters ────────────────────────────────────────────────────

  function emitGateDraw(e: GateDrawEvent) {
    onGateDrawn?.(e)
    activeTool.value = null
  }

  function emitGateEdit(gateId: string, definition: Record<string, unknown>) {
    onGateEdited?.({ gateId, definition })
  }

  // ── Watchers ──────────────────────────────────────────────────────────

  watch(plotData, () => {
    if (!pixi) return
    // Plot data shifts when an inherited plot's parent gate changes — its
    // sample subset can shrink/grow. Drop the locked domain so axes follow the
    // new range. Non-inherited plots only fire this watcher on initial load,
    // where unlocked → locked is the same code path.
    xDomainLocked = false
    yDomainLocked = false
    updateScales()
    renderData()
    renderAxes()
    renderGates()
  }, { deep: true })

  watch(gates, () => {
    renderGates()
  }, { deep: true })

  watch(() => highlightedSampleIds?.value, () => {
    renderData()
  })

  // Watch container for mount
  watch(container, async (el) => {
    if (el) {
      await nextTick()
      destroy()
      await init()
    }
  })

  onBeforeUnmount(() => {
    destroy()
  })

  // ── Public API ────────────────────────────────────────────────────────

  return {
    activeTool,
    isDrawing,
    drawPreviewColor,
    init,
    destroy,
    setOnGateDrawn(cb: (e: GateDrawEvent) => void) { onGateDrawn = cb },
    setOnGateEdited(cb: (e: GateEditEvent) => void) { onGateEdited = cb },
    refresh() {
      updateScales()
      renderData()
      renderAxes()
      renderGates()
    },
  }
}
