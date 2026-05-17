<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ArrowUpRight } from 'lucide-vue-next'
import { usePopoutsStore, type WorkspaceTab } from '@/stores/popouts'
import { usePopoutDrag } from '@/composables/usePopoutDrag'
import PopoutDragOverlay from '@/components/workspace/PopoutDragOverlay.vue'

interface TabSpec { id: WorkspaceTab; name: string; path: string }

const route = useRoute()
const router = useRouter()
const popouts = usePopoutsStore()

const tabs: TabSpec[] = [
  { id: 'gallery', name: 'Gallery', path: '/workspace/gallery' },
  { id: 'classes', name: 'Classes', path: '/workspace/classes' },
  { id: 'clusters', name: 'Clusters', path: '/workspace/clusters' },
  { id: 'analysis', name: 'Analysis', path: '/workspace/analysis' },
  { id: 'segmentation', name: 'Segmentation', path: '/workspace/segmentation' },
]

const activeTabIndex = computed(() => {
  const index = tabs.findIndex(tab => route.path.startsWith(tab.path))
  return index === -1 ? 0 : index
})

const isActiveTabDetached = computed(() => popouts.has(tabs[activeTabIndex.value].id))

const { draggingTab, cursorOutside, canDetach, onDragStart, onDrag, onDragEnd } = usePopoutDrag()

// Brief tint when a tab transitions from attached → detached (just popped out).
const recentlyDetachedId = ref<WorkspaceTab | null>(null)
watch(
  () => [...popouts.detached],
  (next, prev) => {
    const added = next.find(t => !prev?.includes(t))
    if (!added) return
    recentlyDetachedId.value = added
    setTimeout(() => {
      if (recentlyDetachedId.value === added) recentlyDetachedId.value = null
    }, 350)
  },
)

// If the active route's tab is detached, fall back to the first attached tab
// so the main window doesn't render an empty workspace for that tab.
watch(isActiveTabDetached, (detached) => {
  if (!detached) return
  const firstAttached = tabs.find(t => !popouts.has(t.id))
  if (firstAttached) router.replace(firstAttached.path)
}, { immediate: true })

function focusPopout(tabId: WorkspaceTab) {
  window.ipcRenderer?.invoke('popout:focus', tabId)
}

function onTabClick(e: MouseEvent, tab: TabSpec) {
  if (popouts.has(tab.id)) {
    e.preventDefault()
    focusPopout(tab.id)
    return
  }
  router.push(tab.path)
}
</script>

<template>
  <div class="h-full flex flex-col bg-base-200">
    <!-- Tab Bar -->
    <div class="flex border-b-2 border-neutral relative bg-transparent items-end px-6 pt-4 shrink-0">
      <div class="relative flex">
        <!-- Animated sliding rectangle backdrop -->
        <div
          v-show="!isActiveTabDetached"
          class="absolute top-0 bottom-0 left-0 bg-neutral transition-transform duration-300 ease-out z-0 rounded-t-[2px]"
          :style="{ width: '130px', transform: `translateX(${activeTabIndex * 130}px)` }"
        ></div>

        <button
          v-for="(tab, index) in tabs"
          :key="tab.path"
          type="button"
          :draggable="!popouts.has(tab.id) && canDetach"
          :title="popouts.has(tab.id)
            ? `${tab.name} is open in a separate window — click to focus`
            : (canDetach ? `Drag outside the window to detach ${tab.name}` : 'At least one tab must remain in the main window')"
          class="relative w-[130px] py-2.5 text-center font-bold text-[11px] uppercase tracking-widest transition-all duration-300 z-10 select-none flex items-center justify-center gap-1.5"
          :class="[
            popouts.has(tab.id)
              ? 'text-neutral/40 cursor-pointer hover:text-neutral/60'
              : activeTabIndex === index
                ? 'text-base-100'
                : 'text-neutral hover:bg-neutral/5 hover:text-neutral',
            draggingTab?.id === tab.id ? 'opacity-30 scale-95' : '',
            recentlyDetachedId === tab.id ? 'bg-accent/30' : '',
            !popouts.has(tab.id) && canDetach ? 'cursor-grab active:cursor-grabbing' : '',
          ]"
          @click="onTabClick($event, tab)"
          @dragstart="onDragStart($event, tab)"
          @drag="onDrag"
          @dragend="onDragEnd"
        >
          {{ tab.name }}
          <ArrowUpRight v-if="popouts.has(tab.id)" class="w-3 h-3" />
        </button>
      </div>
    </div>

    <!-- Workspace Content Area -->
    <div class="flex-1 overflow-hidden flex flex-col p-4 relative">
      <PopoutDragOverlay
        :visible="!!draggingTab"
        :tab-name="draggingTab?.name ?? ''"
        :cursor-outside="cursorOutside"
      />
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </div>
  </div>
</template>
