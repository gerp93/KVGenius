import { NavLink, Outlet } from 'react-router-dom';

/** Shell for the Library section: a sub-tab strip (Output / Prompts) above the routed page.
 * This div is the scroll container (`.page`) that Output's infinite scroll observes. */
export default function LibraryLayout() {
  return (
    <div className="page library-page">
      <div className="button-row library-tabs">
        <NavLink
          to="/library/output"
          className={({ isActive }) => `library-tabs__link${isActive ? ' active' : ''}`}
        >
          🖼️ Output
        </NavLink>
        <NavLink
          to="/library/prompts"
          className={({ isActive }) => `library-tabs__link${isActive ? ' active' : ''}`}
        >
          📝 Prompts
        </NavLink>
      </div>
      <div className="library-page__body">
        <Outlet />
      </div>
    </div>
  );
}
