<script setup lang="ts">
import { onMounted } from 'vue'
import { RotateCcw } from 'lucide-vue-next'
import { useSettingsStore } from '@/stores/settings'

const store = useSettingsStore()

onMounted(() => {
  store.fetchSettings()
})

function update(field: string, value: unknown) {
  store.updateSettings({ [field]: value })
}
</script>

<template>
  <div class="p-8 max-w-2xl mx-auto flex flex-col h-full">
    <div class="flex items-center justify-between mb-8 pb-4 border-b border-base-300 shrink-0">
      <h1 class="text-2xl font-bold tracking-widest uppercase">Settings</h1>
      <button class="h-9 px-4 flex items-center gap-2 rounded-[2px] hover:bg-base-200 text-[11px] font-bold tracking-widest uppercase transition-colors" @click="store.resetSettings()">
        <RotateCcw class="w-4 h-4 stroke-[1.5px]" /> Reset Defaults
      </button>
    </div>

    <div v-if="store.settings" class="bg-base-100 shadow-sm border border-base-300 rounded-[2px] p-8 space-y-8 overflow-y-auto">
      
      <div class="space-y-2">
        <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Theme</label>
        <select
          class="select select-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral"
          :value="store.settings.theme"
          @change="update('theme', ($event.target as HTMLSelectElement).value)"
        >
          <option value="dark">DARK</option>
          <option value="light">LIGHT</option>
        </select>
      </div>

      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <label class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Thumbnail Quality</label>
          <span class="font-mono text-xs font-bold">{{ store.settings.thumbnail_quality }}</span>
        </div>
        <input
          type="range"
          min="1" max="100"
          :value="store.settings.thumbnail_quality"
          class="range range-xs range-neutral w-full transition-none"
          @change="update('thumbnail_quality', Number(($event.target as HTMLInputElement).value))"
        />
      </div>

      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <label class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Images per Collage</label>
          <span class="font-mono text-xs font-bold">{{ store.settings.images_per_collage }}</span>
        </div>
        <input
          type="range"
          min="1" max="50"
          :value="store.settings.images_per_collage"
          class="range range-xs range-neutral w-full transition-none"
          @change="update('images_per_collage', Number(($event.target as HTMLInputElement).value))"
        />
      </div>

      <div class="space-y-2">
        <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Default Segmentation Method</label>
        <input
          type="text"
          class="input input-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral"
          :value="store.settings.default_segmentation_method"
          @change="update('default_segmentation_method', ($event.target as HTMLInputElement).value)"
        />
      </div>

      <div class="space-y-2">
        <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Default Feature Method</label>
        <input
          type="text"
          class="input input-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral"
          :value="store.settings.default_feature_method"
          @change="update('default_feature_method', ($event.target as HTMLInputElement).value)"
        />
      </div>

      <div class="space-y-2">
        <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Default Dim Reduction Method</label>
        <input
          type="text"
          class="input input-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral"
          :value="store.settings.default_dim_reduction_method"
          @change="update('default_dim_reduction_method', ($event.target as HTMLInputElement).value)"
        />
      </div>
    
    </div>

    <div v-else class="flex justify-center py-16">
      <span class="loading loading-spinner loading-lg text-secondary"></span>
    </div>
  </div>
</template>
