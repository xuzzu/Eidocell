<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useAnalysisStore } from '@/stores/analysis'
import { useSessionStore } from '@/stores/session'
import * as analysisApi from '@/api/analysis'
import type { GateTreeNode } from '@/api/analysis'
import PopulationTreeNode from '@/components/analysis/PopulationTreeNode.vue'

const analysis = useAnalysisStore()
const sessionStore = useSessionStore()

const tree = ref<GateTreeNode[]>([])
const expandedIds = ref<Set<string>>(new Set())
const loading = ref(false)

async function fetchTree() {
  const sid = sessionStore.currentSessionId
  if (!sid) return
  loading.value = true
  try {
    tree.value = await analysisApi.getPopulationTree(sid)
    // Auto-expand all nodes that have children
    function collectIds(nodes: GateTreeNode[]) {
      for (const n of nodes) {
        if (n.children.length > 0) expandedIds.value.add(n.id)
        collectIds(n.children)
      }
    }
    collectIds(tree.value)
  } finally {
    loading.value = false
  }
}

onMounted(() => fetchTree())

// Refresh tree when gates change
watch(() => analysis.gates, () => fetchTree(), { deep: true })

function toggleExpand(id: string) {
  if (expandedIds.value.has(id)) {
    expandedIds.value.delete(id)
  } else {
    expandedIds.value.add(id)
  }
}

async function toggleActive(gateId: string, currentActive: boolean) {
  await analysis.updateGate(gateId, { is_active: !currentActive })
  await fetchTree()
}

function selectGate(gateId: string) {
  // Find which plot this gate belongs to and select it
  function findGate(nodes: GateTreeNode[]): GateTreeNode | null {
    for (const n of nodes) {
      if (n.id === gateId) return n
      const found = findGate(n.children)
      if (found) return found
    }
    return null
  }
  const gate = findGate(tree.value)
  if (gate) {
    analysis.selectPlot(gate.plot_id)
  }
}

function setAsParent(gateId: string) {
  // Toggle: if already selected as parent, clear it
  if (analysis.parentGateId === gateId) {
    analysis.clearParentGate()
  } else {
    analysis.selectParentGate(gateId)
  }
}
</script>

<template>
  <div v-if="tree.length > 0" class="flex flex-col gap-2">
    <div class="flex items-center justify-between">
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Populations</span>
      <button
        v-if="analysis.parentGateId"
        class="text-[9px] font-mono font-bold text-info hover:text-info/80 tracking-wider uppercase"
        @click="analysis.clearParentGate()"
      >
        Clear parent
      </button>
    </div>

    <!-- Active parent indicator -->
    <div
      v-if="analysis.parentGate"
      class="px-2 py-1.5 rounded-[2px] bg-info/10 border border-info/20 flex items-center gap-1.5"
    >
      <span class="w-2 h-2 rounded-full shrink-0" :style="{ backgroundColor: analysis.parentGate.color }"></span>
      <span class="text-[9px] font-mono font-bold text-info truncate">
        Sub-gating within: {{ analysis.parentGate.name }}
      </span>
    </div>

    <div class="flex flex-col">
      <PopulationTreeNode
        v-for="node in tree"
        :key="node.id"
        :node="node"
        :depth="0"
        :expanded-ids="expandedIds"
        :parent-gate-id="analysis.parentGateId"
        @toggle-expand="toggleExpand"
        @toggle-active="toggleActive"
        @select="selectGate"
        @set-parent="setAsParent"
      />
    </div>
  </div>
</template>
