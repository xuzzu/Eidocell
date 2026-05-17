import { ref, computed, onBeforeUnmount } from 'vue'
import { usePopoutsStore, type WorkspaceTab } from '@/stores/popouts'

const TAB_COUNT = 5

interface TabSpec { id: WorkspaceTab; name: string }

export function usePopoutDrag() {
  const popouts = usePopoutsStore()
  const draggingTab = ref<TabSpec | null>(null)
  const cursorOutside = ref(false)

  // True while a drag could result in a successful detach. Used to gate the
  // "min one tab" rule: if only one tab is in the main shell, drag is blocked.
  const canDetach = computed(() => popouts.detached.length + 1 < TAB_COUNT)

  let dragImageEl: HTMLElement | null = null

  function buildDragImage(name: string): HTMLElement {
    const el = document.createElement('div')
    el.textContent = name.toUpperCase()
    el.style.position = 'fixed'
    el.style.top = '-1000px'
    el.style.left = '-1000px'
    el.style.padding = '8px 18px'
    el.style.background = '#1f2937'
    el.style.color = '#fff'
    el.style.fontFamily = 'inherit'
    el.style.fontSize = '11px'
    el.style.fontWeight = '700'
    el.style.letterSpacing = '0.15em'
    el.style.borderRadius = '2px'
    el.style.boxShadow = '0 12px 30px rgba(0,0,0,0.35)'
    el.style.pointerEvents = 'none'
    document.body.appendChild(el)
    return el
  }

  function cleanupDragImage() {
    if (dragImageEl?.parentElement) {
      dragImageEl.parentElement.removeChild(dragImageEl)
    }
    dragImageEl = null
  }

  function isOutsideWindow(screenX: number, screenY: number): boolean {
    const left = window.screenX
    const top = window.screenY
    const right = left + window.outerWidth
    const bottom = top + window.outerHeight
    return screenX < left || screenX > right || screenY < top || screenY > bottom
  }

  function onDragStart(e: DragEvent, tab: TabSpec) {
    if (popouts.has(tab.id)) {
      e.preventDefault()
      return
    }
    if (!canDetach.value) {
      e.preventDefault()
      return
    }
    draggingTab.value = tab
    cursorOutside.value = false
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = 'move'
      e.dataTransfer.setData('application/x-eidocell-tab', tab.id)
      dragImageEl = buildDragImage(tab.name)
      e.dataTransfer.setDragImage(dragImageEl, 60, 16)
    }
  }

  function onDrag(e: DragEvent) {
    if (!draggingTab.value) return
    // screenX/Y are 0,0 in dragend on some browsers; during drag they are valid.
    if (e.screenX === 0 && e.screenY === 0) return
    cursorOutside.value = isOutsideWindow(e.screenX, e.screenY)
  }

  function onDragEnd(e: DragEvent) {
    const tab = draggingTab.value
    draggingTab.value = null
    cleanupDragImage()
    if (!tab) return
    const outside = cursorOutside.value || isOutsideWindow(e.screenX, e.screenY)
    cursorOutside.value = false
    if (!outside) return
    if (!canDetach.value) return
    window.ipcRenderer?.invoke('popout:open', tab.id, e.screenX, e.screenY)
  }

  onBeforeUnmount(() => cleanupDragImage())

  return { draggingTab, cursorOutside, canDetach, onDragStart, onDrag, onDragEnd }
}
