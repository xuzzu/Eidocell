<script setup lang="ts">
import type { Component } from 'vue'

defineProps<{
  icon?: Component
  title: string
  description?: string
  actionLabel?: string
}>()

const emit = defineEmits<{ action: [] }>()
</script>

<template>
  <div class="flex flex-col items-center justify-center py-20 text-neutral/40">
    <component v-if="icon" :is="icon" class="w-16 h-16 mb-6 stroke-[1px]" />
    <h3 class="text-lg font-bold tracking-widest uppercase text-neutral/70">{{ title }}</h3>
    <p v-if="description" class="mt-2 text-sm font-mono tracking-wide">{{ description }}</p>
    <button v-if="actionLabel" class="h-9 px-6 mt-6 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity hover:opacity-80" @click="emit('action')">
      {{ actionLabel }}
    </button>
  </div>
</template>
