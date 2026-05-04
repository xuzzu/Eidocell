import { ref, onMounted, onUnmounted, watch, type Ref } from 'vue'

interface SelectionRect {
  x: number
  y: number
  width: number
  height: number
}

const DRAG_THRESHOLD = 5

const INTERACTIVE_TAGS = new Set(['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A'])

function isInteractive(el: HTMLElement): boolean {
  let node: HTMLElement | null = el
  while (node) {
    if (INTERACTIVE_TAGS.has(node.tagName)) return true
    if (node.getAttribute('role') === 'button') return true
    node = node.parentElement
  }
  return false
}

function rectsIntersect(a: DOMRect, b: DOMRect): boolean {
  return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom)
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(v, hi))
}

export function useDragSelect(
  containerRef: Ref<HTMLElement | null | undefined>,
  cardSelector: string,
  idAttribute: string,
  onSelect: (ids: string[]) => void,
  existingSelection: Ref<Set<string>>,
  triggerRef?: Ref<HTMLElement | null | undefined>,
) {
  const selectionRect = ref<SelectionRect | null>(null)
  const isDragging = ref(false)

  let startX = 0
  let startY = 0
  let dragging = false
  let additiveHeld = false
  let shiftHeld = false

  function onMouseDown(e: MouseEvent) {
    if (e.button !== 0) return
    const target = e.target as HTMLElement
    if (isInteractive(target)) return
    if (target.closest('[data-drag-exclude]')) return

    const container = containerRef.value
    if (!container) return

    // Ignore clicks on the scroll container's scrollbar
    const cRect = container.getBoundingClientRect()
    const insideContainer =
      e.clientX >= cRect.left && e.clientX <= cRect.right &&
      e.clientY >= cRect.top && e.clientY <= cRect.bottom
    if (insideContainer && (e.clientX > cRect.right - 16 || e.clientY > cRect.bottom - 16)) {
      return
    }

    // Don't start a drag from a card (let the card's click handler run)
    if (target.closest(cardSelector)) return

    shiftHeld = e.shiftKey
    additiveHeld = e.ctrlKey || e.metaKey || e.shiftKey

    // Clamp the starting point to the scroll container's bounds so clicks
    // in surrounding margins anchor the rectangle to the nearest edge.
    const clientX = clamp(e.clientX, cRect.left, cRect.right)
    const clientY = clamp(e.clientY, cRect.top, cRect.bottom)
    startX = clientX - cRect.left + container.scrollLeft
    startY = clientY - cRect.top + container.scrollTop
    dragging = false

    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  }

  function onMouseMove(e: MouseEvent) {
    const container = containerRef.value
    if (!container) return

    const cRect = container.getBoundingClientRect()
    const clientX = clamp(e.clientX, cRect.left, cRect.right)
    const clientY = clamp(e.clientY, cRect.top, cRect.bottom)
    const currentX = clientX - cRect.left + container.scrollLeft
    const currentY = clientY - cRect.top + container.scrollTop

    const dx = currentX - startX
    const dy = currentY - startY

    if (!dragging && Math.abs(dx) < DRAG_THRESHOLD && Math.abs(dy) < DRAG_THRESHOLD) {
      return
    }

    dragging = true
    isDragging.value = true

    const x = Math.min(startX, currentX)
    const y = Math.min(startY, currentY)
    const width = Math.abs(dx)
    const height = Math.abs(dy)

    selectionRect.value = { x, y, width, height }
  }

  function onMouseUp(_e: MouseEvent) {
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)

    if (dragging && selectionRect.value) {
      const container = containerRef.value
      if (container) {
        const cards = container.querySelectorAll(cardSelector)
        const containerRect = container.getBoundingClientRect()

        // Convert selection rect from container-relative to viewport-relative
        const selRect = {
          left: selectionRect.value.x - container.scrollLeft + containerRect.left,
          top: selectionRect.value.y - container.scrollTop + containerRect.top,
          right: selectionRect.value.x + selectionRect.value.width - container.scrollLeft + containerRect.left,
          bottom: selectionRect.value.y + selectionRect.value.height - container.scrollTop + containerRect.top,
        } as DOMRect

        const draggedIds: string[] = []
        cards.forEach((card) => {
          const cardRect = card.getBoundingClientRect()
          if (rectsIntersect(selRect, cardRect)) {
            const id = card.getAttribute(idAttribute)
            if (id) draggedIds.push(id)
          }
        })

        if (additiveHeld) {
          // Add drag bounds to existing selection (shift/ctrl/meta all behave additive)
          const merged = new Set(existingSelection.value)
          draggedIds.forEach((id) => merged.add(id))
          onSelect(Array.from(merged))
        } else {
          onSelect(draggedIds)
        }
      }
    } else if (!dragging && !additiveHeld && !shiftHeld) {
      // Background click with no modifiers clears the selection.
      // With shift/ctrl/meta, preserve current selection (matches OS file managers).
      onSelect([])
    }

    selectionRect.value = null
    dragging = false
    isDragging.value = false
  }

  // Listen on `triggerRef` if provided, else on `containerRef`. Re-attach
  // when the underlying element appears or swaps.
  let attachedTo: HTMLElement | null = null
  function getTrigger(): HTMLElement | null {
    return (triggerRef?.value ?? containerRef.value) ?? null
  }
  function syncListener() {
    const next = getTrigger()
    if (next === attachedTo) return
    if (attachedTo) attachedTo.removeEventListener('mousedown', onMouseDown)
    attachedTo = next
    if (next) next.addEventListener('mousedown', onMouseDown)
  }

  onMounted(syncListener)
  watch(() => [containerRef.value, triggerRef?.value], syncListener)

  onUnmounted(() => {
    if (attachedTo) attachedTo.removeEventListener('mousedown', onMouseDown)
    attachedTo = null
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
  })

  return { selectionRect, isDragging }
}
