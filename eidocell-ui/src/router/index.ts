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
            path: '/sessions',
            name: 'sessions',
            component: () => import('../views/SessionsView.vue'),
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
    if (to.path.startsWith('/workspace') || to.path.startsWith('/export')) {
        const session = useSessionStore()
        if (!session.hasCurrentSession) {
            return { name: 'sessions' }
        }
    }
})

export default router
