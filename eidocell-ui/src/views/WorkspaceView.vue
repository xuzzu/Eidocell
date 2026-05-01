<script setup lang="ts">
import { useRoute } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()

const tabs = [
  { name: 'Gallery', path: '/workspace/gallery' },
  { name: 'Classes', path: '/workspace/classes' },
  { name: 'Clusters', path: '/workspace/clusters' },
  { name: 'Analysis', path: '/workspace/analysis' },
  { name: 'Segmentation', path: '/workspace/segmentation' },
]

const activeTabIndex = computed(() => {
  const index = tabs.findIndex(tab => route.path.startsWith(tab.path))
  return index === -1 ? 0 : index
})
</script>

<template>
  <div class="h-full flex flex-col bg-base-200">
    <!-- Tab Bar -->
    <div class="flex border-b-2 border-neutral relative bg-transparent items-end px-6 pt-4 shrink-0">
      <div class="relative flex">
        <!-- Animated sliding rectangle backdrop -->
        <div 
          class="absolute top-0 bottom-0 left-0 bg-neutral transition-transform duration-300 ease-out z-0 rounded-t-[2px]"
          :style="{ width: '130px', transform: `translateX(${activeTabIndex * 130}px)` }"
        ></div>
        
        <router-link
          v-for="(tab, index) in tabs"
          :key="tab.path"
          :to="tab.path"
          class="w-[130px] py-2.5 text-center font-bold text-[11px] uppercase tracking-widest transition-colors duration-300 z-10"
          :class="activeTabIndex === index
            ? 'text-base-100'
            : 'text-neutral hover:bg-neutral/5 hover:text-neutral'"
        >
          {{ tab.name }}
        </router-link>
      </div>
    </div>

    <!-- Workspace Content Area -->
    <div class="flex-1 overflow-hidden flex flex-col p-4">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </div>
  </div>
</template>
