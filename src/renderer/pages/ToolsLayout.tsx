import { Outlet } from 'react-router-dom';

/** Shell for the Tools section (Upscale for now; Image to image and others to follow - each is linked
 * from the top nav bar, like Library's pages). A scrolling page; each tool fills its own body. */
export default function ToolsLayout() {
  return (
    <div className="page tools-page">
      <Outlet />
    </div>
  );
}
