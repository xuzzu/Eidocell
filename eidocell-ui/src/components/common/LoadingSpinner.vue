<script setup lang="ts">
defineProps<{ size?: string }>()
</script>

<template>
  <span class="loading loading-spinner" :class="size ?? 'loading-md'"></span>
</template>
