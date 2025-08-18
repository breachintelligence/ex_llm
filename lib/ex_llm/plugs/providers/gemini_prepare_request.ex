defmodule ExLLM.Plugs.Providers.GeminiPrepareRequest do
  @moduledoc """
  Prepares a request for the Google Gemini API.

  Transforms the generic ExLLM message format into Gemini's expected
  format. This includes:
  - Converting message structure to Gemini's contents format
  - Handling system instructions
  - Formatting multimodal content (text and images)
  - Setting appropriate parameters
  """

  use ExLLM.Plug
  alias ExLLM.Infrastructure.Logger

  @impl true
  def call(%Request{config: config} = request, _opts) do
    body = build_request_body(request)

    # Set the dynamic endpoint based on the model and streaming
    model = config[:model] || "gemini-2.0-flash"
    is_streaming = Map.get(request.options, :stream, false) == true

    endpoint =
      if is_streaming do
        "/models/#{model}:streamGenerateContent?alt=sse"
      else
        "/models/#{model}:generateContent"
      end

    request
    |> Map.put(:provider_request, body)
    |> Request.assign(:http_path, endpoint)
    |> Request.assign(:request_prepared, true)
  end

  defp build_request_body(%Request{messages: messages, config: config}) do
    # Separate system messages
    {system_messages, other_messages} = Enum.split_with(messages, &(&1[:role] == "system"))

    # Build base request
    body = %{
      contents: format_contents(other_messages),
      generationConfig: build_generation_config(config)
    }

    # Add system instruction if present
    body =
      case system_messages do
        [] ->
          body

        [%{content: content} | _rest] ->
          Map.put(body, :systemInstruction, %{
            parts: [%{text: content}]
          })
      end

    # Add safety settings if configured
    body
    |> maybe_add_safety_settings(config)
    |> maybe_add_tools(config)
    |> compact()
  end

  defp format_contents(messages) do
    messages
    |> Enum.map(&format_message/1)
    |> merge_consecutive_same_role()
  end

  defp format_message(%{role: role, content: content}) do
    %{
      "role" => normalize_role(role),
      "parts" => format_parts(content)
    }
  end

  defp normalize_role("user"), do: "user"
  defp normalize_role("assistant"), do: "model"
  defp normalize_role("model"), do: "model"
  defp normalize_role("tool"), do: "function"

  # System handled separately
  defp normalize_role("system"), do: "user"
  defp normalize_role(_), do: "user"

  defp format_parts(content) when is_binary(content) do
    [%{"text" => content}]
  end

  defp format_parts(%{"functionResponse" => _} = content) do
    [content]
  end

  defp format_parts(content) when is_list(content) do
    Enum.map(content, fn
      %{type: "text", text: text} ->
        %{"text" => text}

      %{type: "image", data: data, media_type: media_type} ->
        %{
          "inlineData" => %{
            "mimeType" => media_type,
            "data" => data
          }
        }

      %{type: "image", image: %{data: data, media_type: media_type}} ->
        %{
          "inlineData" => %{
            "mimeType" => media_type,
            "data" => data
          }
        }

      %{type: "image_url", image_url: %{url: url}} ->
        # Convert data URLs to Gemini format
        case parse_data_url(url) do
          {:ok, media_type, data} ->
            %{
              "inlineData" => %{
                "mimeType" => media_type,
                "data" => data
              }
            }

          :error ->
            # Skip non-data URLs as Gemini requires inline data
            nil
        end

      # Handle string keys
      %{"text" => _} = item ->
        item

      %{"inlineData" => _} = item ->
        item

      other ->
        Logger.warning("Unknown content type in Gemini message: #{inspect(other)}")
        nil
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp format_parts(content), do: [%{"text" => to_string(content)}]

  defp merge_consecutive_same_role(messages) do
    # Gemini requires alternating user/model messages
    messages
    |> Enum.reduce([], fn message, acc ->
      case acc do
        [] ->
          [message]

        [%{"role" => last_role} = last | rest] ->
          if last_role == message["role"] do
            # Merge parts
            merged = %{
              "role" => last_role,
              "parts" => last["parts"] ++ message["parts"]
            }

            [merged | rest]
          else
            [message | acc]
          end
      end
    end)
    |> Enum.reverse()
  end

  defp build_generation_config(config) do
    %{}
    |> maybe_add_config(:temperature, config[:temperature])
    |> maybe_add_config(:topP, config[:top_p])
    |> maybe_add_config(:topK, config[:top_k])
    |> maybe_add_config(:maxOutputTokens, config[:max_tokens])
    |> maybe_add_config(:stopSequences, config[:stop_sequences])
    |> maybe_add_config(:candidateCount, config[:candidate_count])
  end

  defp maybe_add_config(map, _key, nil), do: map
  defp maybe_add_config(map, key, value), do: Map.put(map, key, value)

  defp maybe_add_safety_settings(body, %{safety_settings: settings}) when is_list(settings) do
    Map.put(body, :safetySettings, settings)
  end

  defp maybe_add_safety_settings(body, _), do: body

  defp maybe_add_tools(body, %{tools: tools}) when is_list(tools) do
    # Convert to Gemini tool format
    formatted_tools = Enum.map(tools, &format_tool/1)
    Map.put(body, :tools, formatted_tools)
  end

  defp maybe_add_tools(body, _), do: body

  defp format_tool(%{function_declarations: _declarations} = tool) do
    # Already in Gemini format
    tool
  end

  defp format_tool(%{type: "function", function: function}) do
    # Convert from OpenAI format
    %{
      functionDeclarations: [
        %{
          name: function["name"],
          description: function["description"],
          parameters: function["parameters"]
        }
      ]
    }
  end

  defp format_tool(tool) do
    Logger.warning("Unknown tool format: #{inspect(tool)}")
    tool
  end

  defp parse_data_url("data:" <> rest) do
    case String.split(rest, ";base64,", parts: 2) do
      [media_type, data] -> {:ok, media_type, data}
      _ -> :error
    end
  end

  defp parse_data_url(_), do: :error

  defp compact(map) do
    map
    |> Enum.reject(fn {_k, v} -> is_nil(v) or v == %{} end)
    |> Map.new()
  end
end
