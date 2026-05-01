<script setup lang="ts">
import { Folder, Trash2, Image } from 'lucide-vue-next'
import type { SessionListItem } from '@/types'

defineProps<{ session: SessionListItem }>()
const emit = defineEmits<{
  select: [id: string]
  delete: [id: string]
}>()

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}
</script>

<template>
  <div
    class="bg-base-100 border border-base-300 rounded-[2px] hover:border-neutral transition-colors cursor-pointer group flex flex-col h-full"
    @click="emit('select', session.id)"
  >
    <div class="p-5 flex-1 flex flex-col">
      <div class="flex items-start justify-between mb-4">
        <div class="flex items-center gap-4">
          <div class="w-10 h-10 rounded-[2px] border border-neutral/20 bg-base-200 flex items-center justify-center text-neutral group-hover:bg-neutral group-hover:text-base-100 transition-colors shrink-0">
            <Folder class="w-5 h-5 stroke-[1.5px]" />
          </div>
          <div class="min-w-0">
            <h3 class="font-bold text-sm tracking-wider uppercase truncate">{{ session.name }}</h3>
            <p class="text-[10px] font-mono text-neutral/50 truncate mt-0.5" :title="session.images_directory">{{ session.images_directory }}</p>
          </div>
        </div>
        <button
          class="w-8 h-8 shrink-0 flex items-center justify-center text-neutral/30 hover:text-error hover:bg-error/10 rounded-[2px] transition-colors"
          @click.stop="emit('delete', session.id)"
        >
          <Trash2 class="w-4 h-4 stroke-[1.5px]" />
        </button>
      </div>
      <div class="mt-auto pt-4 border-t border-base-200 flex items-center justify-between text-[10px] font-mono text-neutral/60 tracking-tight">
        <span class="flex items-center gap-1.5 font-bold text-neutral">
          <Image class="w-3.5 h-3.5 stroke-[1.5px]" />
          {{ session.sample_count }} SAMPLES
        </span>
        <span>OPENED: {{ formatDate(session.last_opened_at).toUpperCase() }}</span>
      </div>
    </div>
  </div>
</template>
