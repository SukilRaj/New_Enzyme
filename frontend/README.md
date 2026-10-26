# Integrated PET frontend

Initial source copied from `D:\Final_Year_Project\EnzAIme_ML\frontend` on 26 October 2026. This is the maintained local integration copy. Changes add FASTA import, reference-coordinate navigation and export, input-format guidance, and preservation of sequence/structure state across navigation.

From `D:\Enzyme_ML_new`:

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
npm --prefix frontend run lint
node --test frontend/tests/sequenceFiles.test.js
.\Start-Research.cmd
```

The integrated Flask gateway serves `dist/` and supplies the authenticated POST bootstrap. Open port 8765. A standalone Vite development page does not include that bootstrap. The PET service and scientific artifacts remain in the linked project; copying frontend source does not copy its models or data.
