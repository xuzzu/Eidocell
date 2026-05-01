<script setup lang="ts">
import { computed } from 'vue'
import { useSegmentationStore } from '@/stores/segmentation'

const store = useSegmentationStore()

const selectedMethodObj = computed(() =>
  store.methods.find(m => m.id === store.selectedMethod)
)
</script>

<template>
  <div class="flex flex-col gap-6">
    <div class="form-control">
      <label class="label pb-2">
        <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Method</span>
      </label>
      <select
        class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
        :value="store.selectedMethod"
        @change="store.selectMethod(($event.target as HTMLSelectElement).value)"
      >
        <option v-for="m in store.methods" :key="m.id" :value="m.id">{{ m.name }}</option>
      </select>
    </div>

    <div v-if="selectedMethodObj" class="flex flex-col gap-4">
      <div v-for="param in selectedMethodObj.params" :key="param.name" class="form-control">
        <div class="flex items-center justify-between pb-2">
          <label class="label p-0">
            <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">{{ param.label }}</span>
          </label>
          <span class="text-[10px] font-mono text-neutral font-bold">{{ store.params[param.name] ?? param.default }}</span>
        </div>
        <input
          type="range"
          :min="param.min"
          :max="param.max"
          :step="param.step"
          :value="store.params[param.name] ?? param.default"
          class="range range-xs range-neutral flex-1 transition-none"
          @input="store.params[param.name] = Number(($event.target as HTMLInputElement).value)"
        />
        <div class="flex justify-between text-[9px] font-mono text-neutral/40 px-0.5 mt-1">
          <span>{{ param.min }}</span>
          <span>{{ param.max }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
