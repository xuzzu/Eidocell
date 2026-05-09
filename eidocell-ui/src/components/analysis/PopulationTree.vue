<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { RotateCcw, GitMerge } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'
import type { PopulationTreeNode as TreeNode } from '@/types'
import PopulationTreeNode from '@/components/analysis/PopulationTreeNode.vue'
import GateContextMenu from '@/components/analysis/GateContextMenu.vue'
import GateRenameDialog from '@/components/analysis/GateRenameDialog.vue'

const analysis = useAnalysisStore()

const ctxMenu = ref<{
  visible: boolean
  x: number
  y: number
  gateId: string | null
  gateName: string
}>({ visible: false, x: 0, y: 0, gateId: null, gateName: '' })

const renameDialog = ref<InstanceType<typeof GateRenameDialog>>()

function onNodeContextmenu(id: string, name: string, e: MouseEvent) {
  ctxMenu.value = { visible: true, x: e.clientX, y: e.clientY, gateId: id, gateName: name }
}

async function onRename() {
  const id = ctxMenu.value.gateId
  const initial = ctxMenu.value.gateName
  if (!id) return
  const newName = await renameDialog.value?.open(initial)
  if (newName) {
    await analysis.updateGate(id, { name: newName })
  }
}

function onDeleteFromMenu() {
  const id = ctxMenu.value.gateId
  if (id) analysis.deleteGate(id)
}

const expandedIds = ref<Set<string>>(new Set(['__root__']))
const draggingId = ref<string | null>(null)

const tree = computed(() => analysis.populationTree)

const plotNames = computed<Record<string, string>>(() => {
  const map: Record<string, string> = {}
  for (const p of analysis.plots) map[p.id] = p.name
  return map
})

const hierarchicalNodes = computed(() => {
  const flat: TreeNode[] = []
  function walk(n: TreeNode) {
    flat.push(n)
    for (const c of n.children) walk(c)
  }
  if (tree.value?.root) walk(tree.value.root)
  return flat
})

const draggingDescendants = computed<Set<string>>(() => {
  const ids = new Set<string>()
  if (!draggingId.value) return ids
  const start = hierarchicalNodes.value.find(n => n.id === draggingId.value)
  if (!start) return ids
  function collect(n: TreeNode) {
    ids.add(n.id)
    for (const c of n.children) collect(c)
  }
  collect(start)
  return ids
})

onMounted(() => {
  if (!tree.value) analysis.fetchPopulationTree()
})

watch(
  () => tree.value,
  newTree => {
    if (!newTree) return
    function autoExpand(n: TreeNode) {
      if (n.children.length > 0) expandedIds.value.add(n.id)
      for (const c of n.children) autoExpand(c)
    }
    autoExpand(newTree.root)
  },
  { immediate: true },
)

function toggleExpand(id: string) {
  if (expandedIds.value.has(id)) expandedIds.value.delete(id)
  else expandedIds.value.add(id)
}

function selectPopulation(id: string | null) {
  // Treat the synthetic root id as null (clears selection / shows all events).
  analysis.selectPopulation(id === '__root__' ? null : id)
}

function deleteGate(id: string) {
  if (id === '__root__') return
  analysis.deleteGate(id)
}

function onDragStart(id: string) { draggingId.value = id }
function onDragEnd() { draggingId.value = null }

async function onReparent(sourceId: string, newParentId: string | null) {
  draggingId.value = null
  const node = hierarchicalNodes.value.find(n => n.id === sourceId)
  if (!node) return
  const target = newParentId === '__root__' ? null : newParentId
  if ((node.parent_gate_id ?? null) === target) return
  await analysis.updateGate(sourceId, { parent_gate_id: target })
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <div class="flex items-center justify-between">
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Populations</span>
      <button
        v-if="tree && analysis.selectedGateId"
        class="h-6 px-2 rounded-[2px] flex items-center gap-1.5 text-[9px] font-bold tracking-widest uppercase text-neutral/50 hover:bg-neutral/10 hover:text-neutral transition-colors"
        title="Clear population selection (show all events)"
        @click="analysis.resetAllGates()"
      >
        <RotateCcw class="w-3 h-3 stroke-[2px]" />
        Clear
      </button>
    </div>

    <div v-if="!tree" class="text-[10px] font-mono text-neutral/30 tracking-wider">
      Loading…
    </div>

    <div v-else class="flex flex-col">
      <!-- Column header -->
      <div class="flex items-center gap-1 h-5 px-1 mb-1 text-[8px] font-mono font-bold tracking-widest uppercase text-neutral/30 border-b border-base-300">
        <span class="flex-1 min-w-0 ml-7 truncate">Population</span>
        <span class="shrink-0 mr-1">Count / %</span>
      </div>

      <!-- Hierarchical tree, rooted at synthetic 'All Events' -->
      <PopulationTreeNode
        :node="tree.root"
        :depth="0"
        :is-last="true"
        :expanded-ids="expandedIds"
        :plot-names="plotNames"
        :dragging-id="draggingId"
        :dragging-descendants="draggingDescendants"
        :selected-id="analysis.selectedGateId"
        @toggle-expand="toggleExpand"
        @select="selectPopulation"
        @delete="deleteGate"
        @drag-start="onDragStart"
        @drag-end="onDragEnd"
        @reparent="onReparent"
        @contextmenu="onNodeContextmenu"
      />

      <!-- Boolean populations (flat list) -->
      <div v-if="tree.booleans.length > 0" class="mt-3">
        <div class="flex items-center gap-1.5 px-1 pb-1 text-[8px] font-mono font-bold tracking-widest uppercase text-neutral/30 border-b border-base-300">
          <GitMerge class="w-3 h-3 stroke-[2px]" />
          <span>Boolean populations</span>
        </div>

        <div class="flex flex-col mt-1">
          <PopulationTreeNode
            v-for="b in tree.booleans"
            :key="b.id"
            :node="b"
            :depth="0"
            :is-last="true"
            :expanded-ids="expandedIds"
            :plot-names="plotNames"
            :dragging-id="draggingId"
            :dragging-descendants="draggingDescendants"
            :selected-id="analysis.selectedGateId"
            @toggle-expand="toggleExpand"
            @select="selectPopulation"
            @delete="deleteGate"
            @drag-start="onDragStart"
            @drag-end="onDragEnd"
            @reparent="onReparent"
            @contextmenu="onNodeContextmenu"
          />
        </div>
      </div>
    </div>

    <p
      v-if="tree && (tree.root.children.length > 0 || tree.booleans.length > 0)"
      class="text-[9px] font-mono text-neutral/30 leading-snug mt-1"
    >
      Click a row to make it the active population. Drag rows to reparent. Right-click to rename.
    </p>

    <Teleport to="body">
      <GateContextMenu
        v-if="ctxMenu.visible"
        :x="ctxMenu.x"
        :y="ctxMenu.y"
        :can-delete="true"
        @rename="onRename"
        @delete="onDeleteFromMenu"
        @close="ctxMenu.visible = false"
      />
    </Teleport>

    <GateRenameDialog ref="renameDialog" />
  </div>
</template>
