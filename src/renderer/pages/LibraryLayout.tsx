import { Outlet } from 'react-router-dom';

/** Shell for the Library section (Output / Prompts, which are linked from the top nav bar).
 * A fixed-height page whose body Output fills and scrolls internally - see `.library-page`. */
export default function LibraryLayout() {
  return (
    <div className="page library-page">
      <div className="library-page__body">
        <Outlet />
      </div>
    </div>
  );
}
