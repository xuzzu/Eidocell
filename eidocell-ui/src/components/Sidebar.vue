<script setup lang="ts">
import { Home, Folder, Download, Settings } from 'lucide-vue-next'
import { useRoute } from 'vue-router'
import { useSessionStore } from '@/stores/session'

const route = useRoute()
const sessionStore = useSessionStore()

const navItems = [
  { name: 'Workspace', path: '/', icon: Home },
  { name: 'Sessions', path: '/sessions', icon: Folder },
  { name: 'Export', path: '/export', icon: Download },
  { name: 'Settings', path: '/settings', icon: Settings },
]

const isActive = (itemPath: string) => {
  if (itemPath === '/') {
    return route.path === '/' || route.path.startsWith('/workspace');
  }
  return route.path.startsWith(itemPath);
};
</script>

<template>
  <div class="w-[72px] hover:w-60 group bg-base-100 h-screen flex flex-col text-neutral border-r border-base-300 transition-[width] duration-300 ease-in-out z-50 shrink-0 overflow-visible absolute top-0 left-0 hover:shadow-xl md:relative">
    <div class="h-14 flex items-center px-[22px] border-b border-base-300 shrink-0">
      <div class="w-7 h-7 bg-neutral flex items-center justify-center text-neutral-content font-bold text-lg shrink-0 outline outline-1 outline-neutral rounded-[2px]">
        E
      </div>
      <span class="ml-4 font-bold text-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap overflow-hidden text-neutral">EIDOCELL</span>
    </div>

    <nav class="flex-1 py-4 flex flex-col gap-1 px-2">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        class="flex items-center px-3.5 py-2.5 rounded-[2px] transition-all duration-200"
        :class="isActive(item.path)
          ? 'bg-neutral text-neutral-content font-bold'
          : 'text-neutral/60 hover:bg-base-200 hover:text-neutral'"
      >
        <component :is="item.icon" class="w-[22px] h-[22px] shrink-0 stroke-[2px]" />
        <span class="ml-4 text-[11px] font-bold tracking-widest uppercase opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap overflow-hidden">
          {{ item.name }}
        </span>
      </router-link>
    </nav>

    <!-- Session indicator -->
    <div class="px-2 pb-4">
      <router-link
        to="/sessions"
        class="flex items-center gap-3 px-4 py-3 rounded-[2px] hover:bg-base-200 transition-colors"
      >
        <div class="w-1.5 h-1.5 shrink-0 rounded-[1px] outline outline-1 outline-offset-1" :class="sessionStore.hasCurrentSession ? 'bg-green-500 outline-green-500/50' : 'bg-transparent outline-neutral/40'"></div>
        <span class="text-xs font-mono font-bold tracking-wider uppercase opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap overflow-hidden text-neutral/70">
          {{ sessionStore.currentSession?.name ?? 'No session' }}
        </span>
      </router-link>
    </div>
  </div>
</template>
