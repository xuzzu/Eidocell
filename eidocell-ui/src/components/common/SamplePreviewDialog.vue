<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'
import { X } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { sampleThumbnailUrl } from '@/api/gallery'
import type { SampleOut } from '@/types'

const sessionStore = useSessionStore()
const sid = computed(() => sessionStore.currentSessionId!)

const dialogRef = ref<HTMLDialogElement>()
const scrollRef = ref<HTMLElement>()
const title = ref('')
const color = ref<string | null>(null)
const samples = ref<SampleOut[]>([])
const total = ref(0)
const loadingMore = ref(false)

// The caller provides a fetch function for loading more pages
let fetchMore: ((offset: number, limit: number) => Promise<{ items: SampleOut[]; total: number }>) | null = null

const PAGE_SIZE = 100

function open(opts: {
  title: string
  color?: string
  samples: SampleOut[]
  total: number
  fetchMore?: (offset: number, limit: number) => Promise<{ items: SampleOut[]; total: number }>
}) {
  title.value = opts.title
  color.value = opts.color ?? null
  samples.value = opts.samples
  total.value = opts.total
  fetchMore = opts.fetchMore ?? null
  dialogRef.value?.showModal()
  nextTick(() => scrollRef.value?.scrollTo(0, 0))
}

function close() {
  dialogRef.value?.close()
}

const hasMore = computed(() => samples.value.length < total.value)

async function onScroll(e: Event) {
  if (!hasMore.value || loadingMore.value || !fetchMore) return
  const el = e.target as HTMLElement
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 200) {
    loadingMore.value = true
    try {
      const result = await fetchMore(samples.value.length, PAGE_SIZE)
      samples.value.push(...result.items)
      total.value = result.total
    } finally {
      loadingMore.value = false
    }
  }
}

defineExpose({ open, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box max-w-4xl max-h-[80vh]">
      <div class="flex items-center gap-2 mb-4">
        <span
          v-if="color"
          class="w-4 h-4 rounded-full shrink-0"
          :style="{ backgroundColor: color }"
        ></span>
        <h3 class="font-bold text-lg flex-1">{{ title }}</h3>
        <span class="text-sm text-base-content/50">{{ total }} samples</span>
        <button class="btn btn-ghost btn-sm btn-square" @click="close">
          <X class="w-4 h-4" />
        </button>
      </div>
      <div ref="scrollRef" class="overflow-y-auto max-h-[60vh]" @scroll="onScroll">
        <div class="grid grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-2">
          <div
            v-for="sample in samples"
            :key="sample.id"
            class="rounded-lg overflow-hidden border border-base-200"
          >
            <img
              :src="sampleThumbnailUrl(sid, sample.id)"
              :alt="sample.filename"
              class="w-full aspect-square object-cover"
              loading="lazy"
            />
          </div>
        </div>
        <div v-if="loadingMore" class="flex justify-center py-4">
          <span class="loading loading-spinner loading-sm text-neutral/40"></span>
        </div>
        <div v-if="samples.length === 0" class="text-center py-8 text-base-content/40 text-sm">
          No samples
        </div>
      </div>
    </div>
    <form method="dialog" class="modal-backdrop">
      <button @click="close">close</button>
    </form>
  </dialog>
</template>
