'use client';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import type { RunsFilter } from '@/types';
import { Search, X } from 'lucide-react';

interface RunFiltersProps {
  filters: RunsFilter;
  onFiltersChange: (filters: RunsFilter) => void;
  modelTypes: string[];
  frameworks: string[];
}

export function RunFilters({ filters, onFiltersChange, modelTypes, frameworks }: RunFiltersProps) {
  const hasFilters = Object.values(filters).some((v) => v !== undefined && v !== '');

  const clearFilters = () => {
    onFiltersChange({});
  };

  return (
    <div className="flex flex-wrap items-center gap-4">
      {/* Search */}
      <div className="relative flex-1 min-w-[200px]">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search runs..."
          className="pl-9"
          value={filters.model_type || ''}
          onChange={(e) =>
            onFiltersChange({ ...filters, model_type: e.target.value || undefined })
          }
        />
      </div>

      {/* Model Type Filter */}
      <Select
        value={filters.model_type || 'all'}
        onValueChange={(value) =>
          onFiltersChange({ ...filters, model_type: value === 'all' ? undefined : value })
        }
      >
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Model Type" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Models</SelectItem>
          {modelTypes.map((type) => (
            <SelectItem key={type} value={type}>
              {type}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Framework Filter */}
      <Select
        value={filters.framework || 'all'}
        onValueChange={(value) =>
          onFiltersChange({ ...filters, framework: value === 'all' ? undefined : value })
        }
      >
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Framework" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Frameworks</SelectItem>
          {frameworks.map((fw) => (
            <SelectItem key={fw} value={fw}>
              {fw}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Min Accuracy Filter */}
      <Select
        value={filters.min_accuracy?.toString() || 'any'}
        onValueChange={(value) =>
          onFiltersChange({
            ...filters,
            min_accuracy: value === 'any' ? undefined : parseFloat(value),
          })
        }
      >
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Min Accuracy" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="any">Any Accuracy</SelectItem>
          <SelectItem value="0.5">≥ 50%</SelectItem>
          <SelectItem value="0.7">≥ 70%</SelectItem>
          <SelectItem value="0.8">≥ 80%</SelectItem>
          <SelectItem value="0.9">≥ 90%</SelectItem>
        </SelectContent>
      </Select>

      {/* Clear Filters */}
      {hasFilters && (
        <Button variant="ghost" size="sm" onClick={clearFilters}>
          <X className="mr-1 h-4 w-4" />
          Clear
        </Button>
      )}
    </div>
  );
}
