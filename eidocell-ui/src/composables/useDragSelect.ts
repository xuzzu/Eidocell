import { ref, onMounted, onUnmounted, type Ref } from 'vue'

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

export function useDragSelect(
  containerRef: Ref<HTMLElement | null | undefined>,
  cardSelector: string,
  idAttribute: string,
  onSelect: (ids: string[]) => void,
  existingSelection: Ref<Set<string>>,
) {
  const selectionRect = ref<SelectionRect | null>(null)
  const isDragging = ref(false)

  let startX = 0
  let startY = 0
  let dragging = false
  let ctrlHeld = false

  function onMouseDown(e: MouseEvent) {
    if (e.button !== 0) return
    if (isInteractive(e.target as HTMLElement)) return

    const container = containerRef.value
    if (!container) return

    // Ignore clicks on scrollbar
    if (e.clientX > container.getBoundingClientRect().right - 16 || e.clientY > container.getBoundingClientRect().bottom - 16) {
      return
    }

    // Ignore if clicking on a card to avoid conflict with native drag-and-drop
    if ((e.target as HTMLElement).closest(cardSelector)) {
      return
    }

    ctrlHeld = e.ctrlKey || e.metaKey
    const rect = container.getBoundingClientRect()
    startX = e.clientX - rect.left + container.scrollLeft
    startY = e.clientY - rect.top + container.scrollTop
    dragging = false

    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  }

  function onMouseMove(e: MouseEvent) {
    const container = containerRef.value
    if (!container) return

    const rect = container.getBoundingClientRect()
    const currentX = e.clientX - rect.left + container.scrollLeft
    const currentY = e.clientY - rect.top + container.scrollTop

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

        if (ctrlHeld) {
          // Add to existing selection
          const merged = new Set(existingSelection.value)
          draggedIds.forEach((id) => merged.add(id))
          onSelect(Array.from(merged))
        } else {
          onSelect(draggedIds)
        }
      }
    } else if (!dragging && !ctrlHeld) {
      // If we didn't drag and didn't click on a card or interactive element,
      // it means we clicked on the background. We clear the selection.
      onSelect([])
    }

    selectionRect.value = null
    dragging = false
    isDragging.value = false
  }

  onMounted(() => {
    containerRef.value?.addEventListener('mousedown', onMouseDown)
  })

  onUnmounted(() => {
    containerRef.value?.removeEventListener('mousedown', onMouseDown)
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
  })

  return { selectionRect, isDragging }
}
