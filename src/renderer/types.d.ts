import { KVGeniusAPI } from '../shared/types';

declare global {
  interface Window {
    kvgenius: KVGeniusAPI;
  }
}

export {};
