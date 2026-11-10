import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
const env = loadEnv(mode, process.cwd(), 'TRAFFICPULSE_');
const target = process.env.TRAFFICPULSE_API_TARGET || env.TRAFFICPULSE_API_TARGET || 'http://127.0.0.1:8000';
const url = new URL(target);
if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || url.username || url.password) {
  throw new Error('The development API target must be a local HTTP endpoint.');
}
const proxy = { '/api': { target }, '/health': { target } };

return {
  plugins: [react()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
  preview: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
  test: { environment: 'jsdom', setupFiles: './src/test/setup.js', maxWorkers: 1, fileParallelism: false },
};
});
