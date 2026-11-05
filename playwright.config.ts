import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'tests/browser',timeout:30_000,workers:1,use:{baseURL:process.env.PERMIT_BASE_URL||'http://127.0.0.1:8092',viewport:{width:1440,height:1000},trace:'retain-on-failure'},
webServer:process.env.PERMIT_BASE_URL?undefined:{command:'python3 -m http.server 8092 --bind 127.0.0.1 --directory web',url:'http://127.0.0.1:8092',reuseExistingServer:false}});
