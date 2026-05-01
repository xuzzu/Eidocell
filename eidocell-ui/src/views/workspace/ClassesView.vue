<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Plus, Trash2, Tag, ChevronDown, ChevronUp } from 'lucide-vue-next'
import { useClassesStore } from '@/stores/classes'
import { useSessionStore } from '@/stores/session'
import { classPreviewUrl, getClassStatistics, getClassSamples } from '@/api/classes'
import type { ClassStatistics } from '@/types'
import ClassFormDialog from '@/components/classes/ClassFormDialog.vue'
import TrainingPanel from '@/components/classes/TrainingPanel.vue'
import ConfirmDialog from '@/components/common/ConfirmDialog.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import SamplePreviewDialog from '@/components/common/SamplePreviewDialog.vue'

const classesStore = useClassesStore()
const sessionStore = useSessionStore()
const sid = computed(() => sessionStore.currentSessionId!)

const formDialog = ref<InstanceType<typeof ClassFormDialog>>()
const confirmDialog = ref<InstanceType<typeof ConfirmDialog>>()
const previewDialog = ref<InstanceType<typeof SamplePreviewDialog>>()
const deleteTargetId = ref<string | null>(null)

// Cache-bust preview images on class changes
const previewCacheBuster = ref(Date.now())

// Expandable rows
const expandedClassIds = ref<Set<string>>(new Set())
const classStats = ref<Map<string, ClassStatistics>>(new Map())

onMounted(() => {
  classesStore.fetchClasses()
})

async function onCreate(data: { name: string; color: string }) {
  await classesStore.createClass(data)
  previewCacheBuster.value = Date.now()
}

async function onPreview(id: string) {
  await classesStore.selectClass(id)
  const cls = classesStore.classes.find(c => c.id === id)
  if (classesStore.classSamples) {
    previewDialog.value?.open({
      title: cls?.name ?? 'Class',
      color: cls?.color,
      samples: classesStore.classSamples.items,
      total: classesStore.classSamples.total,
      fetchMore: (offset, limit) => getClassSamples(sid.value, id, offset, limit),
    })
  }
}

function onDeleteRequest(id: string) {
  deleteTargetId.value = id
  confirmDialog.value?.open()
}

async function onDeleteConfirm() {
  if (deleteTargetId.value) {
    await classesStore.deleteClass(deleteTargetId.value)
    expandedClassIds.value.delete(deleteTargetId.value)
    classStats.value.delete(deleteTargetId.value)
    deleteTargetId.value = null
    previewCacheBuster.value = Date.now()
  }
}

async function toggleExpand(classId: string) {
  if (expandedClassIds.value.has(classId)) {
    expandedClassIds.value.delete(classId)
  } else {
    expandedClassIds.value.add(classId)
    if (!classStats.value.has(classId)) {
      try {
        const stats = await getClassStatistics(sid.value, classId)
        classStats.value.set(classId, stats)
      } catch {
        // silently fail — expanded row will show no data
      }
    }
  }
}

function formatAttrName(name: string): string {
  return name.replace(/_/g, ' ')
}
</script>

<template>
  <div class="flex h-full gap-6 p-8 bg-base-200">
    <!-- Left: Class Table -->
    <div class="flex-[2] flex flex-col min-w-0 bg-base-100 border border-base-300 shadow-sm rounded-[2px] p-6 text-neutral overflow-hidden">
      <div class="flex items-center justify-between mb-6 shrink-0">
        <h2 class="text-[14px] font-bold tracking-widest uppercase">Class Manager</h2>
        <button class="h-8 px-4 flex items-center gap-2 rounded-[2px] bg-neutral text-neutral-content text-[10px] font-bold tracking-widest uppercase transition-opacity hover:opacity-80 shadow-sm" @click="formDialog?.open()">
          <Plus class="w-3.5 h-3.5 stroke-[2px]" />
          NEW CLASS
        </button>
      </div>

      <div class="flex-1 overflow-y-auto min-h-0 border border-base-200 rounded-[2px]">
        <EmptyState
          v-if="classesStore.classes.length === 0"
          :icon="Tag"
          title="No classes yet"
          description="CREATE CLASSES TO CATEGORIZE YOUR SAMPLES"
        />

        <table v-else class="table table-sm w-full">
          <thead class="sticky top-0 bg-base-100 z-10">
            <tr class="text-[10px] uppercase font-bold tracking-widest text-neutral/50 border-b border-base-300">
              <th class="pl-4 py-3">Preview</th>
              <th class="py-3">Class Name</th>
              <th class="py-3 text-right">Event Count</th>
              <th class="py-3 w-10"></th>
              <th class="pr-4 py-3 w-12"></th>
            </tr>
          </thead>
          <tbody>
            <template v-for="cls in classesStore.classes" :key="cls.id">
              <tr
                class="cursor-pointer hover:bg-neutral/5 transition-colors border-b border-base-200"
                @click="onPreview(cls.id)"
              >
                <td class="pl-4 py-3">
                  <div class="w-10 h-10 p-0.5 border border-base-300 bg-base-200 rounded-[2px] transition-all" :style="{ borderColor: cls.color }">
                    <img
                      :src="classPreviewUrl(sid, cls.id) + '?t=' + previewCacheBuster"
                      class="w-full h-full object-cover rounded-[1px] bg-white"
                      :alt="cls.name"
                    />
                  </div>
                </td>
                <td class="py-3">
                  <div class="flex items-center gap-3">
                    <span class="w-2.5 h-2.5 rounded-[2px]" :style="{ backgroundColor: cls.color }"></span>
                    <span class="font-bold tracking-wider text-xs uppercase" :style="{ color: cls.color }">
                      {{ cls.name }}
                    </span>
                  </div>
                </td>
                <td class="py-3 text-right">
                  <span class="font-mono text-xs font-bold">{{ cls.sample_count }}</span>
                </td>
                <td class="py-3">
                  <button
                    class="w-7 h-7 flex items-center justify-center rounded-[2px] text-neutral/40 hover:text-neutral hover:bg-neutral/10 transition-colors"
                    @click.stop="toggleExpand(cls.id)"
                  >
                    <component :is="expandedClassIds.has(cls.id) ? ChevronUp : ChevronDown" class="w-4 h-4 stroke-[2px]" />
                  </button>
                </td>
                <td class="pr-4 py-3 text-right">
                  <button
                    v-if="cls.name !== 'Uncategorized'"
                    class="w-8 h-8 flex items-center justify-center text-neutral/30 hover:text-error hover:bg-error/10 rounded-[2px] transition-colors ml-auto"
                    @click.stop="onDeleteRequest(cls.id)"
                  >
                    <Trash2 class="w-4 h-4 stroke-[1.5px]" />
                  </button>
                </td>
              </tr>

              <!-- Expanded stats row -->
              <tr v-if="expandedClassIds.has(cls.id)" class="bg-base-200/30 border-b border-base-200">
                <td colspan="5" class="px-6 py-4">
                  <div v-if="!classStats.has(cls.id)" class="flex justify-center py-2">
                    <span class="loading loading-spinner loading-xs text-neutral/40"></span>
                  </div>
                  <div v-else class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                    <div
                      v-for="attr in classStats.get(cls.id)?.attributes"
                      :key="attr.name"
                      class="p-2 bg-base-100 border border-base-300 rounded-[2px]"
                    >
                      <div class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">{{ formatAttrName(attr.name) }}</div>
                      <div v-if="attr.mean != null" class="text-xs font-mono font-bold mt-1">{{ attr.mean.toFixed(2) }}</div>
                      <div v-if="attr.std != null" class="text-[9px] font-mono text-neutral/40">&plusmn; {{ attr.std.toFixed(2) }}</div>
                      <div v-if="attr.mean == null" class="text-[9px] font-mono text-neutral/30 mt-1">N/A</div>
                    </div>
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Right: Training Panel -->
    <TrainingPanel />

    <!-- Dialogs -->
    <ClassFormDialog ref="formDialog" @submit="onCreate" />
    <ConfirmDialog
      ref="confirmDialog"
      title="Delete Class"
      message="Samples in this class will be moved to Uncategorized."
      confirm-label="Delete"
      :danger="true"
      @confirm="onDeleteConfirm"
    />
    <SamplePreviewDialog ref="previewDialog" />
  </div>
</template>
