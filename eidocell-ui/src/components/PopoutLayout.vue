<script setup lang="ts"></script>

<template>
  <div class="flex h-screen overflow-hidden bg-base-200 text-neutral font-sans selection:bg-neutral/20 selection:text-neutral">
    <main class="flex-1 flex flex-col relative overflow-hidden bg-base-200">
      <div class="flex-1 overflow-hidden flex flex-col p-4">
        <slot />
      </div>
    </main>
  </div>
</template>
