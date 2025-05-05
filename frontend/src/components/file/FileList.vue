<template>
  <div class="flex-1 overflow-hidden flex flex-col">
    <h3>업로드된 파일</h3>
    <div class="space-y-2 overflow-y-auto">
      <div 
        v-for="file in files" 
        :key="file.id"
        @click="$emit('select-file', file.id)"
        class="file-item"
        :class="{ selected: selectedFileId === file.id }"
      >
        <div class="file-info">
          <FileText class="w-4 h-4 text-gray-500" />
          <span class="file-name">{{ file.name }}</span>
        </div>
        <button 
          @click.stop="$emit('delete-file', file.id)"
          class="text-gray-400 hover:text-gray-600"
        >
          <X class="w-4 h-4" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { FileText, X } from 'lucide-vue-next'

defineProps({
  files: {
    type: Array,
    required: true
  },
  selectedFileId: {
    type: [String, Number],
    default: null
  }
})

defineEmits(['select-file', 'delete-file'])
</script> 