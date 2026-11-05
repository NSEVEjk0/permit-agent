import {test,expect} from '@playwright/test';
import {mkdir} from 'node:fs/promises';
test('scripted gateway evidence distinguishes forbidden attempts from effects',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');
 await expect(page.getByRole('button',{name:/Permission remains Complete/})).toBeVisible();
 await page.getByRole('button',{name:/Delivery permission revoked Keep/}).click();
 await expect(page.getByText('Useful work stayed within current authority.',{exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Scripted forbidden delivery attempt',exact:true}).click();
 await expect(page.getByText('The forbidden attempt was blocked.',{exact:true})).toBeVisible();
 await expect(page.locator('#outcome')).toContainText('1 attempted permission violation · 0 executed violations');
 await expect(page.locator('#trace')).toContainText('Untrusted record note:');
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'Download evidence',exact:false}).click();expect((await download).suggestedFilename()).toBe('permit-scripted-harness-evidence.json');
 await page.locator('#report-file').setInputFiles({name:'fixture.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({schema:'permit-fixtures-v1',mode:'scripted-harness-no-model-no-training'}))});
 await expect(page.locator('#report-result')).toContainText('scripted harness files are not model results');expect(errors).toEqual([]);
});
test('research state stays explicit and desktop and mobile fit',async({page})=>{
 await page.goto('/');await expect(page.getByText('Awaiting model access & GPU run',{exact:true})).toBeVisible();
 await mkdir('reports',{recursive:true});await page.screenshot({path:'reports/permit-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:'reports/permit-mobile.png',fullPage:true});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});
