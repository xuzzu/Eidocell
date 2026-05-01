<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Play, Trash2, Zap } from 'lucide-vue-next'
import { useLearningStore } from '@/stores/learning'
import { useClassesStore } from '@/stores/classes'

const learning = useLearningStore()
const classesStore = useClassesStore()

const modelName = ref('')
const classifierType = ref<'knn' | 'linear_probe'>('knn')
const knnNeighbors = ref(5)
const linearC = ref(1.0)
const linearMaxIter = ref(1000)

onMounted(() => {
  learning.fetchModels()
})

async function onTrain() {
  if (!modelName.value.trim()) return
  const params: Record<string, unknown> =
    classifierType.value === 'knn'
      ? { n_neighbors: knnNeighbors.value }
      : { C: linearC.value, max_iter: linearMaxIter.value }
  await learning.train(modelName.value.trim(), classifierType.value, params)
}

async function onClassify(modelId: string) {
  await learning.infer(modelId)
  classesStore.fetchClasses()
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>

<template>
  <div class="flex-1 bg-base-100 border border-base-300 shadow-sm rounded-[2px] p-6 flex flex-col gap-6 overflow-y-auto text-neutral">
    <h2 class="text-[14px] font-bold tracking-widest uppercase shrink-0">Training & Classification</h2>

    <!-- Train Section -->
    <div class="space-y-4 border-b border-base-300 pb-6">
      <div class="text-[10px] font-bold tracking-widest uppercase text-neutral/50">Train New Model</div>

      <div class="space-y-2">
        <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Model Name</label>
        <input
          v-model="modelName"
          type="text"
          placeholder="my-model"
          class="input input-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
        />
      </div>

      <div class="space-y-2">
        <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Classifier</label>
        <select
          v-model="classifierType"
          class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
        >
          <option value="knn">KNN</option>
          <option value="linear_probe">Linear Probe</option>
        </select>
      </div>

      <!-- KNN params -->
      <div v-if="classifierType === 'knn'" class="space-y-2">
        <div class="flex items-center justify-between">
          <label class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">N Neighbors</label>
          <span class="font-mono text-xs font-bold">{{ knnNeighbors }}</span>
        </div>
        <input
          v-model.number="knnNeighbors"
          type="range"
          min="1"
          max="50"
          class="range range-xs range-neutral w-full transition-none"
        />
      </div>

      <!-- Linear Probe params -->
      <template v-if="classifierType === 'linear_probe'">
        <div class="space-y-2">
          <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Regularization (C)</label>
          <input
            v-model.number="linearC"
            type="number"
            min="0.01"
            max="100"
            step="0.1"
            class="input input-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
          />
        </div>
        <div class="space-y-2">
          <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Max Iterations</label>
          <input
            v-model.number="linearMaxIter"
            type="number"
            min="100"
            max="10000"
            step="100"
            class="input input-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
          />
        </div>
      </template>

      <!-- Train button -->
      <button
        class="h-10 w-full rounded-[2px] flex items-center justify-center gap-2 text-[11px] font-bold tracking-widest uppercase transition-opacity"
        :class="(modelName.trim() && !learning.training)
          ? 'bg-neutral text-neutral-content hover:opacity-80'
          : 'bg-base-200 text-neutral/40 cursor-not-allowed'"
        :disabled="!modelName.trim() || learning.training"
        @click="onTrain"
      >
        <span v-if="learning.training" class="loading loading-spinner loading-sm"></span>
        <Play v-else class="w-4 h-4 stroke-[2px]" />
        {{ learning.training ? 'Training...' : 'Train Model' }}
      </button>
    </div>

    <!-- Training result -->
    <div
      v-if="learning.lastTrainResult"
      class="p-4 bg-success/10 border border-success/20 rounded-[2px]"
    >
      <div class="text-[10px] font-bold tracking-widest uppercase text-success-content">Training Complete</div>
      <div class="text-xs font-mono mt-1">
        Accuracy: {{ learning.lastTrainResult.accuracy != null ? (learning.lastTrainResult.accuracy * 100).toFixed(1) + '%' : 'N/A' }}
      </div>
      <div class="text-[10px] font-mono text-neutral/60 mt-0.5">
        {{ learning.lastTrainResult.n_classes }} classes · {{ learning.lastTrainResult.n_training_samples }} samples
      </div>
    </div>

    <!-- Error -->
    <div v-if="learning.error" class="p-3 bg-error/10 border border-error/20 rounded-[2px] text-xs font-mono text-error">
      {{ learning.error }}
    </div>

    <!-- Inference result -->
    <div
      v-if="learning.lastInferenceResult"
      class="p-4 bg-info/10 border border-info/20 rounded-[2px]"
    >
      <div class="text-[10px] font-bold tracking-widest uppercase">Classification Complete</div>
      <div class="text-xs font-mono mt-1">{{ learning.lastInferenceResult.total_predicted }} samples classified</div>
      <div
        v-for="(count, className) in learning.lastInferenceResult.assignments"
        :key="String(className)"
        class="text-[10px] font-mono text-neutral/60"
      >
        {{ className }}: {{ count }}
      </div>
    </div>

    <!-- Models List -->
    <div class="space-y-3">
      <div class="text-[10px] font-bold tracking-widest uppercase text-neutral/50">Trained Models</div>

      <div
        v-if="learning.models.length === 0"
        class="text-xs font-mono text-neutral/40 py-4 text-center"
      >
        NO MODELS YET
      </div>

      <div
        v-for="model in learning.models"
        :key="model.id"
        class="p-4 border border-base-300 rounded-[2px] space-y-3"
      >
        <div class="flex items-center justify-between">
          <span class="font-bold text-xs uppercase tracking-wider">{{ model.name }}</span>
          <button
            class="w-7 h-7 flex items-center justify-center rounded-[2px] text-neutral/30 hover:text-error hover:bg-error/10 transition-colors"
            @click="learning.removeModel(model.id)"
          >
            <Trash2 class="w-3.5 h-3.5 stroke-[1.5px]" />
          </button>
        </div>
        <div class="flex gap-4 text-[10px] font-mono text-neutral/60">
          <span class="uppercase">{{ model.classifier_type }}</span>
          <span>{{ model.accuracy != null ? (model.accuracy * 100).toFixed(1) + '%' : 'N/A' }}</span>
          <span>{{ model.n_classes }} classes</span>
        </div>
        <div class="text-[9px] font-mono text-neutral/40">
          {{ formatDate(model.created_at) }}
        </div>
        <button
          class="h-8 w-full rounded-[2px] flex items-center justify-center gap-2 text-[10px] font-bold tracking-widest uppercase transition-opacity"
          :class="learning.inferring
            ? 'bg-base-200 text-neutral/40 cursor-not-allowed'
            : 'bg-neutral text-neutral-content hover:opacity-80'"
          :disabled="learning.inferring"
          @click="onClassify(model.id)"
        >
          <span v-if="learning.inferring" class="loading loading-spinner loading-xs"></span>
          <Zap v-else class="w-3.5 h-3.5 stroke-[2px]" />
          Classify Uncategorized
        </button>
      </div>
    </div>
  </div>
</template>
