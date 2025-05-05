<template>
  <div class="flex-1">
    <div v-if="data" class="visualization">
      <!-- 시각화 컴포넌트가 여기에 들어갈 예정 -->
      <pre class="text-sm">{{ JSON.stringify(data, null, 2) }}</pre>
    </div>
    <div v-else class="text-gray-500 text-sm flex items-center gap-2">
      <BarChart class="w-4 h-4" />
      분석 결과가 여기에 표시됩니다
    </div>
  </div>
</template>

<script setup>
import { BarChart } from 'lucide-vue-next'

defineProps({
  data: {
    type: Object,
    default: null
  }
})
</script> 