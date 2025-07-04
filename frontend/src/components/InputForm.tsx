import { useState } from "react";
import { Button } from "@/components/ui/button";
import { SquarePen, StopCircle, ArrowUpCircleIcon } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import React from "react";

// Updated InputFormProps
interface InputFormProps {
  onSubmit: (inputValue: string) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasHistory: boolean;
}

export const InputForm: React.FC<InputFormProps> = ({
  onSubmit,
  onCancel,
  isLoading,
  hasHistory,
}) => {
  const [internalInputValue, setInternalInputValue] = useState("");

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue);
    setInternalInputValue("");
  };

  const handleInternalKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleInternalSubmit();
    }
  };

  const isSubmitDisabled = !internalInputValue.trim() || isLoading;

  return (
    <form
      onSubmit={handleInternalSubmit}
      className={`flex flex-col gap-2 p-3 `}
    >
        <div
        className={`flex flex-row items-center justify-between text-white rounded-full break-words min-h-7 bg-neutral-700 px-4 pt-3 `}
        >
            <Textarea
            value={internalInputValue}
            onChange={(e) => setInternalInputValue(e.target.value)}
            onKeyDown={handleInternalKeyDown}
            placeholder="Latest Updates on the Big Beautiful Bill"
            className={`w-full text-neutral-100 placeholder-neutral-500 resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none 
                            md:text-base  min-h-[56px] max-h-[200px]`}
            rows={1}
            />
            <div className="-mt-3">
            {isLoading ? (
                <Button
                type="button"
                variant="ghost"
                size="icon"
                className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 cursor-pointer rounded-full transition-all duration-200"
                onClick={onCancel}
                >
                <StopCircle className="h-5 w-5" />
                </Button>
            ) : (
                <Button
                type="submit"
                variant="ghost"
                className={`${
                    isSubmitDisabled
                    ? "text-neutral-500"
                    : "text-blue-500 hover:text-blue-400 hover:bg-blue-500/10"
                } p-2 cursor-pointer rounded-full transition-all duration-200 text-base`}
                disabled={isSubmitDisabled}
                >
                <ArrowUpCircleIcon className="h-8 w-8" />
                </Button>
            )}
            </div>
            {hasHistory && (
            <Button
                className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer rounded-full"
                variant="default"
                onClick={() => window.location.reload()}
            >
                <SquarePen size={16} />
                New Search
            </Button>
            )}
        </div>
    </form>
  );
};
