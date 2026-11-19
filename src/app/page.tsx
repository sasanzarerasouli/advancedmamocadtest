import Analyzer from "@/components/Analyzer";

export default function HomePage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Mammography CAD – executable pipeline</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-600">
          Upload a mammogram (e.g. the RMLO image from your request) and the pipeline described in the Methodology tab runs on its pixels: breast/pectoral/skin/nipple/marker
          segmentation → exclusion map → scale-space mass localisation + Fuzzy-C-Means boundary → white-top-hat calcification detection + DBSCAN clustering. Nothing is uploaded to a server;
          only a numeric summary can be stored.
        </p>
      </div>
      <Analyzer />
    </div>
  );
}
