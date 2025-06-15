import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";

export default function App() {
    const [processedEventsTimeline, setProcessedEventsTimeline] = useState<ProcessedEvent[]>([]);
    const [historicalActivities, setHistoricalActivities] = useState<Record<string, ProcessedEvent[]>>({});
    const scrollAreaRef = useRef<HTMLDivElement>(null);
    const hasFinalizeEventOccurredRef = useRef(false);

    const thread = useStream<{
        messages: Message[];        
    }>({
        apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "agent",
    messagesKey: "messages",
    onFinish: (event: any) => {
      console.log(event);
    },
    onUpdateEvent: (event: any) => {
        let processedEvent: ProcessedEvent | null = null;
        if (event.preprocess_input){
            processedEvent = {
                title: "Preprocessing",
                data: "Your query has been refined.",
            }
        } else if (event.extract_filters){
            const filtersExtracted = event.extract_filters.filters || "N/A";
            const stateFound = filtersExtracted.state || "N/A";
            const yearFound = filtersExtracted.year || "N/A";
            const billIdentifierFound = filtersExtracted.bill_identifier || "N/A";
            const filtersData = [];
            if (yearFound !== "N/A") filtersData.push(`Year: ${yearFound}`);
            if (billIdentifierFound !== "N/A") filtersData.push(`Bill: ${billIdentifierFound}`);
            if (stateFound !== "N/A") filtersData.push(`State: ${stateFound}`);

            const dataMessage = filtersData.length > 0 
                ? filtersData.join(", ") 
                : `${Object.keys(filtersExtracted).length} keywords found`;

            processedEvent = {
                title: "Filtering",
                data: dataMessage,
            }
        } else if (event.retrieve_documents){
            const docs = event.retrieve_documents.retrieved_docs || [];
            const numDocs = docs.length;
            processedEvent = {
                title: "Retrieving",
                data: `${numDocs} documents retrieved`,
            }
        } else if (event.grade_documents){
            const docs = event.grade_documents.grade_details || [];
            const numDocs = docs.length;
            processedEvent = {
                title: "Grading",
                data: `Found ${numDocs} relevant documents to your query`,
            }
        } else if (event.reconstruct_full_text){
            processedEvent = {
                title: "Reconstructing",
                data: "Reconstructing bill..."
            }
        } else if (event.summarize_bills){
            processedEvent = {
                title: "Summarizing",
                data: "Summarizing bill..."
            }
        } else if (event.compile_final_research){
            processedEvent = {
                title: "Finalizing",
                data: "Composing and presenting the final answer.",
            };
            hasFinalizeEventOccurredRef.current = true; // not sure if we want this here because we'll render the result cards afterwards
        } 
        if (processedEvent) {
            setProcessedEventsTimeline((prevEvents) => [
                ...prevEvents,
                processedEvent!,
            ]);
        }
    }
    });

    // Scroll to bottom of chat when new message is added
    useEffect(() => {
        if (scrollAreaRef.current) {
            const scrollViewport = scrollAreaRef.current.querySelector(
              "[data-radix-scroll-area-viewport]"
            );
            if (scrollViewport) {
              scrollViewport.scrollTop = scrollViewport.scrollHeight;
            }
          }
        }, [thread.messages]);     
        
    // Update historical activities when finalize event occurs. 
    // TODO: Figure out proper logic for determining the final event
    useEffect(() => {
        if (
            hasFinalizeEventOccurredRef.current &&
            !thread.isLoading &&
            thread.messages.length > 0
        ) {
            const lastMessage = thread.messages[thread.messages.length - 1];
            if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
                setHistoricalActivities((prev) => ({
                    ...prev,
                    [lastMessage.id!]: [...processedEventsTimeline],
                }));
            }
            hasFinalizeEventOccurredRef.current = false;
        }
    }, [thread.messages, thread.isLoading, processedEventsTimeline]);

    const handleSubmit = useCallback(
        (submittedInputValue: string) => {
            if (!submittedInputValue.trim()) return;
            setProcessedEventsTimeline([]);
            hasFinalizeEventOccurredRef.current = false;

            const newMessages: Message[] = [
                ...(thread.messages || []),
                {
                    type: "human",
                    content: submittedInputValue,
                    id: Date.now().toString(),
                },
            ];
            console.log("newMessages", newMessages);
            thread.submit({
                messages: newMessages,
            });
        },
        [thread]
    );

    const handleCancel = useCallback(() => {
        thread.stop();
        window.location.reload();
    }, [thread]);

    return (
        <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
          <main className="flex-1 flex flex-col overflow-hidden max-w-4xl mx-auto w-full">
            <div
              className={`flex-1 overflow-y-auto ${
                thread.messages.length === 0 ? "flex" : ""
              }`}
            >
              {thread.messages.length === 0 ? (
                <WelcomeScreen
                  handleSubmit={handleSubmit}
                  isLoading={thread.isLoading}
                  onCancel={handleCancel}
                />
              ) : (
                <ChatMessagesView
                  messages={thread.messages}
                  isLoading={thread.isLoading}
                  scrollAreaRef={scrollAreaRef}
                  onSubmit={handleSubmit}
                  onCancel={handleCancel}
                  liveActivityEvents={processedEventsTimeline}
                  historicalActivities={historicalActivities}
                />
              )}
            </div>
          </main>
        </div>
      );
}
    