import { createRouter, createWebHashHistory } from 'vue-router'
import { useSessionStore } from '@/stores/session'

const router = createRouter({
    history: createWebHashHistory(import.meta.env.BASE_URL),
    routes: [
        {
            path: '/',
            name: 'workspace',
            component: () => import('../views/WorkspaceView.vue'),
            redirect: '/workspace/gallery',
            children: [
                {
                    path: 'workspace/gallery',
                    name: 'gallery',
                    component: () => import('../views/workspace/GalleryView.vue'),
                },
                {
                    path: 'workspace/classes',
                    name: 'classes',
                    component: () => import('../views/workspace/ClassesView.vue'),
                },
                {
                    path: 'workspace/clusters',
                    name: 'clusters',
                    component: () => import('../views/workspace/ClustersView.vue'),
                },
                {
                    path: 'workspace/segmentation',
                    name: 'segmentation',
                    component: () => import('../views/workspace/SegmentationView.vue'),
                },
                {
                    path: 'workspace/analysis',
                    name: 'analysis',
                    component: () => import('../views/workspace/AnalysisView.vue'),
                },
            ]
        },
        {
            path: '/popout/gallery',
            name: 'popout-gallery',
            component: () => import('../views/workspace/GalleryView.vue'),
            meta: { layout: 'popout', tabId: 'gallery' },
        },
        {
            path: '/popout/classes',
            name: 'popout-classes',
            component: () => import('../views/workspace/ClassesView.vue'),
            meta: { layout: 'popout', tabId: 'classes' },
        },
        {
            path: '/popout/clusters',
            name: 'popout-clusters',
            component: () => import('../views/workspace/ClustersView.vue'),
            meta: { layout: 'popout', tabId: 'clusters' },
        },
        {
            path: '/popout/segmentation',
            name: 'popout-segmentation',
            component: () => import('../views/workspace/SegmentationView.vue'),
            meta: { layout: 'popout', tabId: 'segmentation' },
        },
        {
            path: '/popout/analysis',
            name: 'popout-analysis',
            component: () => import('../views/workspace/AnalysisView.vue'),
            meta: { layout: 'popout', tabId: 'analysis' },
        },
        {
            path: '/sessions',
            name: 'sessions',
            component: () => import('../views/SessionsView.vue'),
        },
        {
            path: '/import/:sessionId',
            name: 'import-wizard',
            component: () => import('../views/ImportWizardView.vue'),
            props: true,
        },
        {
            path: '/export',
            name: 'export',
            component: () => import('../views/ExportView.vue'),
        },
        {
            path: '/settings',
            name: 'settings',
            component: () => import('../views/SettingsView.vue'),
        },
        {
            path: '/:pathMatch(.*)*',
            name: 'not-found',
            redirect: '/sessions',
        },
    ]
})

router.beforeEach((to) => {
    // Popouts don't redirect — they hydrate their session from localStorage /
    // IPC sync. Until then the workspace view just renders empty.
    if (to.path.startsWith('/popout')) return
    if (to.path.startsWith('/workspace') || to.path.startsWith('/export')) {
        const session = useSessionStore()
        if (!session.hasCurrentSession) {
            return { name: 'sessions' }
        }
    }
})

export default router
