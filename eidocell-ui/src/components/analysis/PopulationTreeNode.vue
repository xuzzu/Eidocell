<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  ChevronRight, ChevronDown, Trash2, GripVertical,
  FolderTree, GitMerge, Square, Pentagon, Circle, Crosshair, ArrowLeftRight,
} from 'lucide-vue-next'
import type { PopulationTreeNode as TreeNode } from '@/types'

const props = defineProps<{
  node: TreeNode
  depth: number
  isLast: boolean
  expandedIds: Set<string>
  plotNames: Record<string, string>
  draggingId: string | null
  draggingDescendants: Set<string>
  selectedId: string | null
}>()

const emit = defineEmits<{
  toggleExpand: [id: string]
  select: [id: string]
  delete: [id: string]
  dragStart: [id: string]
  dragEnd: []
  reparent: [sourceId: string, newParentId: string | null]
  contextmenu: [id: string, name: string, e: MouseEvent]
  clearRebound: [id: string]
}>()

const isRoot = computed(() => props.node.gate_type === 'root')
const isBoolean = computed(() => props.node.gate_type === 'boolean')
const isHierarchical = computed(() => !isRoot.value && !isBoolean.value)
const isRebound = computed(() => Boolean(props.node.rebound_at))
const hasChildren = computed(() => props.node.children.length > 0)
const isExpanded = computed(() => props.expandedIds.has(props.node.id))
const isBeingDragged = computed(() => props.draggingId === props.node.id)
const isSelected = computed(() => {
  if (props.node.id === '__root__') return props.selectedId === null
  return props.selectedId === props.node.id
})

const TypeIcon = computed(() => {
  if (isRoot.value) return FolderTree
  if (isBoolean.value) return GitMerge
  switch (props.node.gate_type) {
    case 'rectangular': return Square
    case 'polygon': return Pentagon
    case 'ellipse': return Circle
    case 'quadrant': return Crosshair
    case 'interval': return ArrowLeftRight
    default: return Square
  }
})

const isDraggable = computed(() => isHierarchical.value)

// Booleans cannot be drop targets; root accepts everything; hierarchical accepts
// non-self, non-descendant gates. Boolean rows are never themselves dragged
// because isDraggable is false on them.
const canAcceptDrop = computed(() => {
  if (!props.draggingId) return false
  if (isBoolean.value) return false
  if (props.draggingId === props.node.id) return false
  if (props.draggingDescendants.has(props.node.id)) return false
  return true
})

const isDropTarget = ref(false)

function onDragStart(e: DragEvent) {
  if (!isDraggable.value) {
    e.preventDefault()
    return
  }
  if (!e.dataTransfer) return
  e.dataTransfer.effectAllowed = 'move'
  e.dataTransfer.setData('application/x-gate-id', props.node.id)
  emit('dragStart', props.node.id)
}

function onDragEnd() {
  emit('dragEnd')
  isDropTarget.value = false
}

function onDragOver(e: DragEvent) {
  if (!canAcceptDrop.value) return
  e.preventDefault()
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'move'
  isDropTarget.value = true
}

function onDragLeave() { isDropTarget.value = false }

function onDrop(e: DragEvent) {
  if (!canAcceptDrop.value) return
  e.preventDefault()
  e.stopPropagation()
  isDropTarget.value = false
  if (props.draggingId) {
    const targetId = isRoot.value ? null : props.node.id
    emit('reparent', props.draggingId, targetId)
  }
}

const countLabel = computed(() => {
  const c = props.node.sample_count
  if (c >= 1000) return (c / 1000).toFixed(c >= 10000 ? 0 : 1) + 'k'
  return c.toString()
})
const fullCount = computed(() => props.node.sample_count.toLocaleString())
</script>

<template>
  <div>
    <div
      class="pop-tree-row flex items-center gap-1 h-7 rounded-[2px] group transition-all duration-150 cursor-pointer overflow-hidden"
      :class="[
        isDropTarget ? 'bg-success/15 ring-1 ring-success/40' : '',
        isSelected ? 'bg-neutral/10 ring-1 ring-neutral/30' : (isDropTarget ? '' : 'hover:bg-base-200/60'),
        isBeingDragged ? 'opacity-30 scale-[0.98]' : '',
      ]"
      :style="{ paddingLeft: `${depth * 12 + 2}px` }"
      :draggable="isDraggable"
      :title="`${node.name} • ${fullCount} • ${node.percentage.toFixed(1)}%`"
      @click.stop="emit('select', node.id)"
      @contextmenu.prevent="!isRoot && emit('contextmenu', node.id, node.name, $event)"
      @dragstart="onDragStart"
      @dragend="onDragEnd"
      @dragover="onDragOver"
      @dragleave="onDragLeave"
      @drop="onDrop"
    >
      <!-- Drag handle (hidden on root + booleans) -->
      <div
        v-if="isDraggable"
        class="flex items-center w-2.5 shrink-0 text-neutral/15 group-hover:text-neutral/40 transition-colors cursor-grab active:cursor-grabbing"
      >
        <GripVertical class="w-2.5 h-2.5 stroke-[2px]" />
      </div>
      <div v-else class="w-2.5 shrink-0"></div>

      <!-- Expand/collapse chevron -->
      <button
        v-if="hasChildren"
        class="w-3 h-7 flex items-center justify-center shrink-0 text-neutral/30 hover:text-neutral transition-colors"
        @click.stop="emit('toggleExpand', node.id)"
      >
        <ChevronDown v-if="isExpanded" class="w-2.5 h-2.5 stroke-[2px]" />
        <ChevronRight v-else class="w-2.5 h-2.5 stroke-[2px]" />
      </button>
      <div v-else class="w-3 shrink-0"></div>

      <!-- Type icon + color dot -->
      <div class="flex items-center gap-1 shrink-0">
        <component
          :is="TypeIcon"
          class="w-3 h-3 stroke-[2px]"
          :class="isRoot ? 'text-neutral/60' : (isBoolean ? 'text-purple-500' : 'text-neutral/50')"
        />
        <span
          v-if="!isRoot"
          class="w-1.5 h-1.5 rounded-full border border-black/10"
          :style="{ backgroundColor: node.color }"
        ></span>
      </div>

      <!-- Name + operator badge for booleans -->
      <span
        class="flex-1 min-w-0 text-[10px] font-mono font-bold truncate"
        :class="isSelected ? 'text-neutral' : 'text-neutral/80'"
      >{{ node.name
        }}<span
          v-if="isBoolean && node.operator"
          class="ml-1 px-1 text-[8px] font-bold tracking-widest rounded-[2px] bg-purple-100 text-purple-700"
        >{{ node.operator }}</span></span>

      <!-- Rebound badge: axis change retargeted this gate's params. Click to dismiss. -->
      <button
        v-if="isRebound && isHierarchical"
        class="ml-1 px-1 h-3.5 flex items-center text-[8px] font-bold tracking-widest rounded-[2px] bg-amber-100 text-amber-700 hover:bg-amber-200 transition-colors shrink-0"
        title="Gate axes were rebound after a plot parameter change. Click to dismiss."
        @click.stop="emit('clearRebound', node.id)"
      >REBOUND</button>

      <!-- Count + % (compact) -->
      <div class="flex items-center gap-1 text-[9px] font-mono text-neutral/40 shrink-0 mr-0.5">
        <span>{{ countLabel }}</span>
        <span class="text-neutral/30">{{ node.percentage.toFixed(0) }}%</span>
      </div>

      <!-- Delete (hidden on root, hover-only) -->
      <button
        v-if="!isRoot"
        class="h-7 w-5 flex items-center justify-center text-neutral/20 hover:text-error transition-colors shrink-0 opacity-0 group-hover:opacity-100"
        title="Delete gate"
        @click.stop="emit('delete', node.id)"
      >
        <Trash2 class="w-3 h-3 stroke-[2px]" />
      </button>
    </div>

    <template v-if="hasChildren && isExpanded">
      <div class="relative" :style="{ marginLeft: `${depth * 12 + 8}px` }">
        <div class="absolute left-0 top-0 bottom-0 w-px bg-base-300"></div>

        <PopulationTreeNode
          v-for="(child, idx) in node.children"
          :key="child.id"
          :node="child"
          :depth="depth + 1"
          :is-last="idx === node.children.length - 1"
          :expanded-ids="expandedIds"
          :plot-names="plotNames"
          :dragging-id="draggingId"
          :dragging-descendants="draggingDescendants"
          :selected-id="selectedId"
          @toggle-expand="(id) => emit('toggleExpand', id)"
          @select="(id) => emit('select', id)"
          @delete="(id) => emit('delete', id)"
          @drag-start="(id) => emit('dragStart', id)"
          @drag-end="emit('dragEnd')"
          @reparent="(s, p) => emit('reparent', s, p)"
          @contextmenu="(id, name, e) => emit('contextmenu', id, name, e)"
          @clear-rebound="(id) => emit('clearRebound', id)"
        />
      </div>
    </template>
  </div>
</template>

<script lang="ts">
export default { name: 'PopulationTreeNode' }
</script>
