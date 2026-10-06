import { Outlet, useLocation } from 'react-router-dom';

/** Shell for the Library section (Output / Prompts, which are linked from the top nav bar).
 * A fixed-height page whose body Output fills and scrolls internally - see `.library-page`. */
export default function LibraryLayout() {
  // Output and Prompts have collapsible side panels that sit flush against the window edge.
  const { pathname } = useLocation();
  const edgeToEdge = pathname.endsWith('/output') || pathname.endsWith('/prompts');
  return (
    <div className={`page library-page${edgeToEdge ? ' library-page--edge' : ''}`}>
      <div className="library-page__body">
        <Outlet />
      </div>
    </div>
  );
}
