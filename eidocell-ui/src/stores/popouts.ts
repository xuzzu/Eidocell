import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export type WorkspaceTab = 'gallery' | 'classes' | 'clusters' | 'segmentation' | 'analysis'

export const usePopoutsStore = defineStore('popouts', () => {
  // Stored as an array (not Set) so the windowSync plugin can serialize it
  // without an adapter. Order is insignificant.
  const detached = ref<WorkspaceTab[]>([])

  const detachedSet = computed(() => new Set(detached.value))

  function has(tabId: WorkspaceTab): boolean {
    return detachedSet.value.has(tabId)
  }

  function add(tabId: WorkspaceTab) {
    if (!detachedSet.value.has(tabId)) {
      detached.value = [...detached.value, tabId]
    }
  }

  function remove(tabId: WorkspaceTab) {
    if (detachedSet.value.has(tabId)) {
      detached.value = detached.value.filter(t => t !== tabId)
    }
  }

  function set(tabs: WorkspaceTab[]) {
    detached.value = [...tabs]
  }

  return { detached, detachedSet, has, add, remove, set }
})
