<template>
  <div class="file-meta">
    <h3>File Information</h3>
    <div v-if="file" class="meta-content">
      <div class="meta-item">
        <span class="label">Name</span>
        <span class="value">{{ file.name }}</span>
      </div>
      <div class="meta-item">
        <span class="label">Size</span>
        <span class="value">{{ formatFileSize(file.size) }}</span>
      </div>
      <div class="meta-item">
        <span class="label">Type</span>
        <span class="value">{{ file.type }}</span>
      </div>
      <div class="meta-item">
        <span class="label">Uploaded</span>
        <span class="value">{{ formatDate(file.uploaded_at) }}</span>
      </div>
    </div>
    <div v-else class="no-file">
      <Info class="w-4 h-4" />
      <span>No file selected</span>
    </div>
  </div>
</template>

<script setup>
import { Info } from 'lucide-vue-next'

const props = defineProps({
  file: {
    type: Object,
    default: null
  }
})

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString()
}
</script>

<style scoped>
.file-meta {
  padding: 1.5rem;
}

.file-meta h3 {
  margin-bottom: 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #64748b;
}

.meta-content {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.meta-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.meta-item .label {
  color: #64748b;
}

.meta-item .value {
  font-weight: 500;
  color: #1e293b;
}

.no-file {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #64748b;
  font-size: 0.875rem;
}
</style> 