import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
import sqlmodel as sq
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:
    from requests.packages.urllib3.util.retry import Retry

from ..context import ctx
from ..dao import Chapter, Novel
from ..assets.languages import language_codes

logger = logging.getLogger(__name__)


class ApiService:
    """Service for uploading novels to external API"""

    def __init__(self) -> None:
        self.session = requests.Session()
        # Configure retry strategy - only retry on actual server failures
        # Don't retry on client errors (400, 401, 403, 404, 413) - these are permanent failures
        # Only retry on server errors (500, 502, 503, 504) and network/timeout issues
        # Reduced retries to 2 to avoid unnecessary attempts
        retry_strategy = Retry(
            total=2,  # Only 2 retries (reduced from default 3 to minimize unnecessary retries)
            backoff_factor=ctx.config.api.retry_delay,  # Use configured delay (default: 5)
            status_forcelist=[500, 502, 503, 504],  # Only retry on server errors
            allowed_methods=["POST"],
            # Don't retry on:
            # - Client errors (400, 401, 403, 404, 413) - permanent failures
            # - Success (200) - no retry needed
            # Only retry on server errors which might be transient
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _get_language_name(self, language_code: Optional[str]) -> str:
        """Convert language code to full language name"""
        if not language_code:
            return "Unknown"
        return language_codes.get(language_code, language_code.capitalize())

    def _determine_status(self, novel: Novel) -> str:
        """Determine novel status based on available data.

        Priority:
        1. Use status stored in `novel.extra['status']` if valid.
        2. If missing and the source site supports status detection (e.g. Asianovel),
           re-scrape the novel page to infer the status and persist it.
        3. Fallback to \"ongoing\".
        """
        # 1) Check novel.extra for status information
        if hasattr(novel, "extra") and isinstance(novel.extra, dict):
            status = str(novel.extra.get("status", "")).lower()
            if status in ["ongoing", "completed", "hiatus"]:
                return status

        # 2) Site-specific fallback: Asianovel / Wuxiasky
        try:
            domain = (getattr(novel, "domain", "") or "").lower()
        except Exception:  # noqa: BLE001
            domain = ""

        url_lower = (getattr(novel, "url", "") or "").lower()
        cover_hint = (getattr(novel, "cover_url", "") or "").lower()

        is_asianovel_source = any(
            hint in value
            for hint in ("asianovel.net", "wuxiasky.net")
            for value in (domain, url_lower, cover_hint)
        )

        if is_asianovel_source:
            logger.warning(
                "Attempting Asianovel status inference for novel '%s' "
                "(domain='%s', url='%s', cover_url='%s')",
                getattr(novel, "title", url_lower or domain),
                domain,
                url_lower,
                cover_hint,
            )
            try:
                inferred = self._infer_status_from_asianovel(novel.url)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to infer status from Asianovel for '%s': %s",
                    getattr(novel, "title", novel.url),
                    e,
                    exc_info=True,
                )
                inferred = None

            if inferred in ["ongoing", "completed", "hiatus"]:
                logger.warning(
                    "Inferred status '%s' for Asianovel novel '%s' via direct HTML parsing",
                    inferred,
                    novel.title,
                )
                # Persist inferred status into novel.extra for future uploads
                try:
                    with ctx.db.session() as sess:
                        db_novel = sess.get(Novel, novel.id)
                        if db_novel:
                            extra = dict(db_novel.extra) if isinstance(db_novel.extra, dict) else {}
                            extra["status"] = inferred
                            sess.exec(
                                sq.update(Novel)
                                .where(Novel.id == novel.id)
                                .values(extra=extra)
                            )
                            sess.commit()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to persist inferred status '%s' for novel '%s': %s",
                        inferred,
                        novel.title,
                        e,
                        exc_info=True,
                    )

                return inferred

        # 3) Default to \"ongoing\" if no status information is available
        return "ongoing"

    def _infer_status_from_asianovel(self, url: str) -> Optional[str]:
        """Infer novel status directly from an Asianovel / Wuxiasky novel page.

        Looks for the status badge:
            <span class=\"story__meta-item story__status _completed\">Completed</span>

        Returns one of: \"ongoing\", \"completed\", \"hiatus\", or None if unknown.
        """
        if not url:
            return None

        logger.debug("Fetching Asianovel page to infer status: %s", url)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Primary: dedicated status span (support both story__ and story_ naming;
        # site uses story_meta-item, story_status_completed)
        status_spans = []
        for sel in (
            "span.story__meta-item.story__status",
            "span.story_meta-item.story_status_completed",
            "span.story_meta-item.story_status_ongoing",
            "span.story_meta-item[class*='story_status']",
        ):
            status_spans = soup.select(sel)
            if status_spans:
                break
        for tag in status_spans:
            if not isinstance(tag, Tag):
                continue
            classes = tag.get("class", [])
            class_str = " ".join(classes)
            raw_text = tag.get_text(strip=True)
            text = raw_text.lower()
            logger.debug(
                "Asianovel infer-status span: classes='%s', text='%s', lower='%s'",
                class_str,
                raw_text,
                text,
            )

            if any("_completed" in c for c in classes):
                return "completed"
            if any("_ongoing" in c for c in classes):
                return "ongoing"

            if "complete" in text or "finished" in text:
                return "completed"
            if "ongoing" in text or "on-going" in text or "on going" in text:
                return "ongoing"
            if "hiatus" in text or "paused" in text:
                return "hiatus"

        # Fallback: any element with a class containing '_completed'
        completed_badge = soup.select_one("[class*='_completed']")
        if isinstance(completed_badge, Tag):
            logger.debug(
                "Asianovel infer-status via '_completed' class on element: %s",
                completed_badge.get_text(strip=True),
            )
            return "completed"

        # Fallback: button or element with data-status="Completed"
        for el in soup.select("[data-status]"):
            if isinstance(el, Tag):
                data_status = (el.get("data-status") or "").strip().lower()
                if "complete" in data_status or "finished" in data_status:
                    return "completed"
                if "ongoing" in data_status or "on-going" in data_status:
                    return "ongoing"
                if "hiatus" in data_status or "paused" in data_status:
                    return "hiatus"

        return None

    def _get_cover_url(self, novel: Novel) -> str:
        """Get the best available cover image URL"""
        # Use the original cover URL from the source website
        # This is the most reliable as it's publicly accessible
        if novel.cover_url:
            return novel.cover_url
        
        # No cover available
        logger.warning(f"Novel '{novel.title}' has no cover URL")
        return ""

    def _format_novel_data(self, novel: Novel, include_chapters: bool = True, only_new: bool = False) -> Dict:
        """Format novel data according to API requirements
        
        Args:
            novel: Novel to format
            include_chapters: Whether to include chapters in the payload
            only_new: If True, only include chapters that haven't been uploaded to API yet
        """
        # Get all chapters for the novel
        chapters = ctx.chapters.list(novel_id=novel.id, is_crawled=True)
        chapters.sort(key=lambda c: c.serial)
        
        # Filter out already-uploaded chapters if only_new is True
        if only_new:
            original_count = len(chapters)
            chapters = [
                c for c in chapters 
                if not (hasattr(c, 'extra') and isinstance(c.extra, dict) and c.extra.get('api_uploaded') is True)
            ]
            new_count = len(chapters)
            if original_count > new_count:
                logger.info(f"Filtered {original_count - new_count} already-uploaded chapters, {new_count} new chapters to upload")

        # Determine status - ensure it's properly extracted
        status = self._determine_status(novel)
        status_source = novel.extra.get('status', 'not found') if hasattr(novel, 'extra') and isinstance(novel.extra, dict) else 'no extra dict'
        logger.warning(
            f"Novel '{novel.title}' status: {status} "
            f"(extracted from: {status_source}, extra keys: {list(novel.extra.keys()) if hasattr(novel, 'extra') and isinstance(novel.extra, dict) else 'N/A'})"
        )

        # Get cover URL
        cover_url = self._get_cover_url(novel)
        if cover_url:
            logger.warning(f"Novel '{novel.title}' cover URL: {cover_url}")
        else:
            logger.warning(f"Novel '{novel.title}' has no cover image URL")

        # Format genres (using tags as genres) - define before logging
        # Capitalize genre names (e.g., "school life" -> "School Life")
        raw_genres = novel.tags or []
        genres = [genre.title() if genre else genre for genre in raw_genres] if raw_genres else []
        
        # Format novel object (no tags field - genres are at root level)
        novel_data = {
            "title": novel.title,
            "author": novel.authors or "Unknown",
            "description": novel.synopsis or "",
            "coverImageUrl": cover_url,
            "status": status,
            "originalLanguage": self._get_language_name(novel.language),
            "sourceUrl": novel.url,
        }
        
        # Log the actual payload being sent
        logger.warning(
            f"Novel data payload for '{novel.title}': "
            f"status='{novel_data['status']}', "
            f"coverImageUrl='{novel_data['coverImageUrl']}', "
            f"genres={genres} (count: {len(genres)})"
        )
        if genres:
            logger.warning(f"Genres being sent: {genres}")
        else:
            logger.warning(f"WARNING: No genres/tags found for novel '{novel.title}'. Novel.tags = {novel.tags}")

        # Add optional fields if available
        # Note: originalTitle and translator are not in the Novel model
        # You may need to extract these from novel.extra or other sources
        if hasattr(novel, "original_title") and novel.original_title:
            novel_data["originalTitle"] = novel.original_title
        if hasattr(novel, "translator") and novel.translator:
            novel_data["translator"] = novel.translator

        # Format chapters (only if include_chapters is True)
        formatted_chapters = []
        if include_chapters:
            for chapter in chapters:
                if not chapter.is_available:
                    logger.warning(f"Chapter {chapter.serial} ({chapter.title}) is not available, skipping")
                    continue

                try:
                    content = ctx.files.load_text(chapter.content_file)
                except Exception as e:
                    logger.error(f"Failed to load chapter {chapter.serial}: {e}")
                    continue

                chapter_data = {
                    "title": chapter.title,
                    "content": content,
                    # API expects snake_case `chapter_number`
                    "chapter_number": chapter.serial,
                }

                # Add publishedAt if available
                if hasattr(chapter, "created_at") and chapter.created_at:
                    # Convert timestamp (milliseconds) to ISO 8601 format
                    if isinstance(chapter.created_at, (int, float)):
                        # Timestamp is in milliseconds, convert to seconds
                        timestamp_ms = chapter.created_at
                        timestamp_s = timestamp_ms / 1000.0 if timestamp_ms > 1e10 else timestamp_ms
                        dt = datetime.fromtimestamp(timestamp_s)
                    else:
                        dt = chapter.created_at
                    chapter_data["publishedAt"] = dt.isoformat() + "Z"

                formatted_chapters.append(chapter_data)

        return {
            "novel": novel_data,
            "genres": genres,
            "chapters": formatted_chapters,
        }
    
    def _format_chapters_batch(self, chapters: List[Chapter], batch_size: int = 50) -> tuple[List[List[Dict]], List[List[Chapter]]]:
        """Format chapters into batches for upload
        
        Returns:
            Tuple of (batches, chapter_objects_list) where chapter_objects_list matches batches
        """
        batches = []
        chapter_objects_list = []
        current_batch = []
        current_chapters = []
        
        for chapter in chapters:
            if not chapter.is_available:
                logger.warning(f"Chapter {chapter.serial} ({chapter.title}) is not available, skipping")
                continue

            try:
                content = ctx.files.load_text(chapter.content_file)
            except Exception as e:
                logger.error(f"Failed to load chapter {chapter.serial}: {e}")
                continue

            chapter_data = {
                "title": chapter.title,
                "content": content,
                "chapter_number": chapter.serial,
            }

            # Add publishedAt if available
            if hasattr(chapter, "created_at") and chapter.created_at:
                if isinstance(chapter.created_at, (int, float)):
                    timestamp_ms = chapter.created_at
                    timestamp_s = timestamp_ms / 1000.0 if timestamp_ms > 1e10 else timestamp_ms
                    dt = datetime.fromtimestamp(timestamp_s)
                else:
                    dt = chapter.created_at
                chapter_data["publishedAt"] = dt.isoformat() + "Z"

            current_batch.append(chapter_data)
            current_chapters.append(chapter)
            
            # When batch is full, add it to batches and start a new one
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                chapter_objects_list.append(current_chapters)
                current_batch = []
                current_chapters = []
        
        # Add remaining chapters as final batch
        if current_batch:
            batches.append(current_batch)
            chapter_objects_list.append(current_chapters)
        
        return batches, chapter_objects_list

    def _mark_chapters_uploaded_objects(self, chapters: List[Chapter]) -> None:
        """Mark chapter objects as uploaded to the external API in their extra field"""
        from ..context import ctx
        if not chapters:
            return
        
        with ctx.db.session() as sess:
            updated_count = 0
            for chapter in chapters:
                if not chapter or not hasattr(chapter, 'id') or not chapter.id:
                    continue
                    
                # Refresh the chapter from the session to ensure we have the latest version
                try:
                    # Merge the chapter into the session
                    chapter = sess.merge(chapter)
                    if not hasattr(chapter, 'extra') or not isinstance(chapter.extra, dict):
                        chapter.extra = {}
                    chapter.extra['api_uploaded'] = True
                    sess.add(chapter)
                    updated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to mark chapter {getattr(chapter, 'id', 'unknown')} as uploaded: {e}")
                    continue
            
            if updated_count > 0:
                sess.commit()
                logger.debug(f"Marked {updated_count}/{len(chapters)} chapters as uploaded to API")

    def _mark_chapters_uploaded(self, chapter_ids: List[str]) -> None:
        """Mark chapters as uploaded to the external API in their extra field (by ID)"""
        from ..context import ctx
        import sqlmodel as sa
        
        # Filter out None/empty IDs
        valid_ids = [cid for cid in chapter_ids if cid and isinstance(cid, str) and len(cid.strip()) > 0]
        
        if not valid_ids:
            logger.warning("No valid chapter IDs provided for marking as uploaded")
            return
        
        with ctx.db.session() as sess:
            # Use bulk update with SQLModel
            from ..dao import Chapter
            chapters = sess.exec(
                sa.select(Chapter).where(Chapter.id.in_(valid_ids))
            ).all()
            
            updated_count = 0
            for chapter in chapters:
                if not hasattr(chapter, 'extra') or not isinstance(chapter.extra, dict):
                    chapter.extra = {}
                chapter.extra['api_uploaded'] = True
                sess.add(chapter)
                updated_count += 1
            
            if updated_count > 0:
                sess.commit()
                logger.debug(f"Marked {updated_count}/{len(valid_ids)} chapters as uploaded to API")

    def _retry_missing_chapters(self, novel: Novel, user_id: Optional[str]) -> None:
        """
        Try to download chapters that failed previously.

        This is useful in cases where a FULL_NOVEL job completed but some
        chapters were not crawled successfully. We try to fetch only the
        missing ones before uploading to the external API.

        Tracks persistent retry attempts per chapter to prevent infinite retries.
        Chapters that have already been retried 3 times are skipped.
        """
        if not user_id:
            return

        # List all chapters for this novel
        chapters = ctx.chapters.list(novel_id=novel.id)
        missing = [chap for chap in chapters if not chap.is_available]
        if not missing:
            return

        max_retries = 3

        # Filter out chapters that have already exceeded max retries
        chapters_to_retry = []
        skipped_chapters = []
        
        for chapter in missing:
            # Get current retry count from chapter.extra
            retry_attempts = chapter.extra.get('api_retry_attempts', 0)
            if retry_attempts >= max_retries:
                skipped_chapters.append((chapter, retry_attempts))
            else:
                chapters_to_retry.append((chapter, retry_attempts))

        if skipped_chapters:
            logger.info(
                "Skipping %d chapters that have already exceeded %d retry attempts for novel '%s' (%s)",
                len(skipped_chapters),
                max_retries,
                novel.title,
                novel.id,
            )
            for chapter, attempts in skipped_chapters:
                logger.debug(
                    "Chapter %s (serial=%d) already retried %d times, skipping",
                    chapter.id,
                    chapter.serial,
                    attempts,
                )

        if not chapters_to_retry:
            if skipped_chapters:
                logger.info("All missing chapters have already exceeded retry limit")
            return

        logger.info(
            "Retrying %d missing chapters for novel '%s' (%s)",
            len(chapters_to_retry),
            novel.title,
            novel.id,
        )

        for chapter, previous_attempts in chapters_to_retry:
            # Calculate how many attempts we can still make
            remaining_attempts = max_retries - previous_attempts
            if remaining_attempts <= 0:
                continue

            success = False
            for attempt in range(1, remaining_attempts + 1):
                current_attempt = previous_attempts + attempt
                try:
                    logger.info(
                        "Retrying chapter %s (serial=%d, title=%s) [attempt %d/%d (total: %d/%d)]",
                        chapter.id,
                        chapter.serial,
                        chapter.title,
                        attempt,
                        remaining_attempts,
                        current_attempt,
                        max_retries,
                    )
                    # This will download and persist the chapter content
                    ctx.crawler.fetch_chapter(user_id, chapter.id)
                    # Refresh chapter state
                    refreshed = ctx.chapters.get(chapter.id)
                    if refreshed.is_available:
                        success = True
                        # Reset retry count on success
                        self._update_chapter_retry_count(chapter.id, 0)
                        logger.info(
                            "Successfully retried chapter %s (serial=%d)",
                            chapter.id,
                            chapter.serial,
                        )
                        break
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "Failed to retry chapter %s (serial=%d) on attempt %d/%d (total: %d/%d): %s",
                        chapter.id,
                        chapter.serial,
                        attempt,
                        remaining_attempts,
                        current_attempt,
                        max_retries,
                        e,
                    )
                    # Update retry count in database after each failed attempt
                    self._update_chapter_retry_count(chapter.id, current_attempt)

            if not success:
                logger.warning(
                    "Giving up on chapter %s (serial=%d) after %d total attempts; it will be skipped in upload",
                    chapter.id,
                    chapter.serial,
                    max_retries,
                )

    def _update_chapter_retry_count(self, chapter_id: str, retry_count: int) -> None:
        """
        Update the API retry attempt count for a chapter in the database.
        
        Args:
            chapter_id: ID of the chapter to update
            retry_count: New retry count to store
        """
        try:
            with ctx.db.session() as sess:
                chapter = sess.get(Chapter, chapter_id)
                if not chapter:
                    logger.warning(f"Chapter {chapter_id} not found when updating retry count")
                    return
                
                # Update the extra field with retry count
                extra = dict(chapter.extra) if chapter.extra else {}
                extra['api_retry_attempts'] = retry_count
                
                # Update the chapter in the database
                sess.exec(
                    sq.update(Chapter)
                    .where(Chapter.id == chapter_id)
                    .values(extra=extra)
                )
                sess.commit()
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Failed to update retry count for chapter {chapter_id}: {e}",
                exc_info=True,
            )

    def upload_novel(self, novel_id: str, user_id: Optional[str] = None, force_upload: bool = False) -> Optional[Dict]:
        """
        Upload a novel to the API endpoint.

        Args:
            novel_id: ID of the novel to upload
            user_id: Optional user ID for retrying missing chapters
            force_upload: If True, allow upload even if auto-upload is disabled (for manual uploads)

        Returns:
            Response data if successful, None otherwise
        """
        if not force_upload and not ctx.config.api.enabled:
            logger.debug("API upload is disabled")
            return None

        if not ctx.config.api.token:
            logger.warning("API token is not configured, skipping upload")
            return None

        # Get novel from database
        novel = ctx.novels.get(novel_id)
        if not novel:
            logger.error(f"Novel {novel_id} not found")
            return None

        logger.warning(f"Preparing to upload novel: {novel.title}")
        logger.warning(f"API URL: {ctx.config.api.url}")
        logger.warning(f"API Token configured: {'Yes' if ctx.config.api.token else 'No'}")

        # Try to re-fetch any missing chapters before formatting data
        try:
            self._retry_missing_chapters(novel, user_id)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Error while retrying missing chapters for novel '%s': %s",
                novel.title,
                e,
                exc_info=True,
            )

        # Format data - for ongoing novels, only upload new chapters
        status = self._determine_status(novel)
        only_new = (status == "ongoing")  # For ongoing novels, only upload new chapters
        
        try:
            data = self._format_novel_data(novel, only_new=only_new)
        except Exception as e:
            logger.error(f"Failed to format novel data: {e}", exc_info=True)
            return None

        # Check if we have chapters
        if not data["chapters"]:
            if only_new:
                logger.info(f"No new chapters to upload for ongoing novel '{novel.title}' (all chapters already uploaded)")
            else:
                logger.warning(f"No chapters available for novel {novel.title}, skipping upload")
            return None

        if only_new:
            logger.info(f"Uploading {len(data['chapters'])} new chapters for ongoing novel '{novel.title}'")
        else:
            logger.info(f"Uploading novel '{novel.title}' with {len(data['chapters'])} chapters")

        # Prepare request with browser-like headers to avoid Vercel Security Checkpoint
        base_url = ctx.config.api.url.split("/api")[0] if "/api" in ctx.config.api.url else ctx.config.api.url.rsplit("/", 1)[0]
        headers = {
            "Authorization": f"Bearer {ctx.config.api.token}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": base_url,
            "Referer": base_url + "/",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

        # Make request with retries
        try:
            # Log the actual payload structure (without chapter content for brevity)
            payload_summary = {
                "novel": {
                    "title": data["novel"]["title"],
                    "author": data["novel"]["author"],
                    "status": data["novel"]["status"],
                    "coverImageUrl": data["novel"]["coverImageUrl"],
                    "description": data["novel"]["description"][:100] + "..." if len(data["novel"]["description"]) > 100 else data["novel"]["description"],
                },
                "chapters_count": len(data.get("chapters", [])),
                "genres_count": len(data.get("genres", [])),
                "genres": data.get("genres", []),  # Include actual genres in summary
            }
            logger.warning(f"API Request Payload Summary: {payload_summary}")
            logger.warning(f"DEBUG: Full genres data being sent: {data.get('genres')}")
            logger.warning(f"Full novel data being sent - status: '{data['novel']['status']}', coverImageUrl: '{data['novel']['coverImageUrl']}'")
            logger.warning(f"Sending POST request to: {ctx.config.api.url}")
            logger.warning(f"Request headers: Authorization={'Bearer ***' if headers.get('Authorization') else 'None'}, Content-Type={headers.get('Content-Type')}")
            
            import json
            # Log a sample of the actual JSON being sent (first 2000 chars)
            json_str = json.dumps(data, indent=2)
            logger.warning(f"JSON Payload (first 2000 chars): {json_str[:2000]}")
            
            # Specifically log genres to verify they're in the payload
            logger.warning(f"DEBUG: Payload structure - genres type: {type(data.get('genres'))}, genres value: {data.get('genres')}")
            logger.warning(f"DEBUG: Payload structure - has 'genres' key: {'genres' in data}, genres count: {len(data.get('genres', []))}")
            
            # Log the exact structure being sent
            if 'genres' not in data:
                logger.error(f"ERROR: 'genres' key is missing from payload! Available keys: {list(data.keys())}")
            elif not data.get('genres'):
                logger.error(f"ERROR: 'genres' key exists but is empty! Novel.tags = {novel.tags}")
            else:
                logger.warning(f"SUCCESS: Genres are in payload: {data.get('genres')}")
            
            # First, make a direct request without retries to see the actual error
            direct_session = requests.Session()
            try:
                direct_response = direct_session.post(
                    ctx.config.api.url,
                    headers=headers,
                    json=data,
                    timeout=300,
                )
                logger.warning(f"Direct API Response Status: {direct_response.status_code}")
                if direct_response.status_code != 200:
                    try:
                        error_json = direct_response.json()
                        logger.error(f"Direct API Error Response: {json.dumps(error_json, indent=2)}")
                    except:
                        logger.error(f"Direct API Error Response (text): {direct_response.text[:2000]}")
            except Exception as direct_e:
                logger.error(f"Direct request error: {direct_e}")
            
            # Now make the request with retries
            response = self.session.post(
                ctx.config.api.url,
                headers=headers,
                json=data,
                timeout=300,  # 5 minute timeout for large uploads
            )
            
            logger.warning(f"API Response Status: {response.status_code}")
            logger.warning(f"API Response Headers: {dict(response.headers)}")
            
            # Check for Vercel Security Checkpoint or rate limiting (429)
            response_text = response.text if response.text else ""
            if response.status_code == 429 or "Vercel Security Checkpoint" in response_text or "data-astro-cid" in response_text:
                if response.status_code == 429:
                    logger.error(
                        f"Rate limit (429) from Vercel when uploading novel '{novel.title}'. "
                        f"Too many requests - Vercel is rate limiting your API calls."
                    )
                    logger.error(
                        f"To fix this: "
                        f"(1) Add a delay between requests, "
                        f"(2) Reduce the number of concurrent uploads, "
                        f"(3) Whitelist your crawler's IP in Vercel, or "
                        f"(4) Upgrade your Vercel plan for higher rate limits."
                    )
                else:
                    logger.error(
                        f"Vercel Security Checkpoint blocked the request to {ctx.config.api.url}. "
                        f"This means Vercel detected the request as automated/bot traffic. "
                        f"Response status: {response.status_code}"
                    )
                    logger.error(
                        f"To fix this, you may need to: "
                        f"(1) Whitelist your crawler's IP address in Vercel, "
                        f"(2) Disable Vercel Security Checkpoint for your API route, "
                        f"(3) Or use a different authentication method that bypasses the checkpoint."
                    )
                # Log a sample of the checkpoint HTML
                checkpoint_sample = response_text[:500] if response_text else "(empty)"
                logger.error(f"Response sample: {checkpoint_sample}")
                return None
            
            # Always log the response body for debugging
            try:
                response_json = response.json()
                logger.warning(f"API Response JSON: {json.dumps(response_json, indent=2)}")
            except:
                response_text_sample = response_text[:2000] if response_text else "(empty)"
                logger.warning(f"API Response Body (text, first 2000 chars): {response_text_sample}")

            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"Successfully uploaded novel '{novel.title}': "
                    f"Novel ID: {result.get('novelId')}, "
                    f"URL: {result.get('novelUrl')}, "
                    f"Chapters imported: {result.get('chaptersImported', len(data['chapters']))}"
                )
                # Mark uploaded chapters (for ongoing novels, track which ones were uploaded)
                if only_new:
                    # Get chapter IDs that were uploaded (those not marked as uploaded)
                    chapters = ctx.chapters.list(novel_id=novel.id, is_crawled=True)
                    chapter_ids_to_mark = [
                        c.id for c in chapters 
                        if not (hasattr(c, 'extra') and isinstance(c.extra, dict) and c.extra.get('api_uploaded') is True)
                    ]
                    if chapter_ids_to_mark:
                        self._mark_chapters_uploaded(chapter_ids_to_mark)
                
                # Add novel status to the result
                novel_status = self._determine_status(novel)
                result['status'] = novel_status
                return result
            elif response.status_code == 413:
                # Payload too large - try batch upload
                logger.warning(f"Payload too large (413) for novel '{novel.title}' with {len(data['chapters'])} chapters. Attempting batch upload...")
                batch_result = self._upload_novel_in_batches(novel, user_id, headers)
                if batch_result:
                    # Add novel status to the batch result
                    novel_status = self._determine_status(novel)
                    batch_result['status'] = novel_status
                return batch_result
            elif response.status_code == 400:
                error_data = response.json()
                logger.error(f"Validation error uploading novel '{novel.title}': {error_data.get('error', 'Unknown error')}")
                return None
            elif response.status_code == 401:
                logger.error("Authentication failed: Invalid API token")
                return None
            elif response.status_code == 429:
                logger.error(
                    f"Rate limit (429) when uploading novel '{novel.title}'. "
                    f"Too many requests - please add a delay between uploads or reduce concurrency."
                )
                return None
            elif response.status_code == 500:
                error_text = response.text[:2000] if response.text else "No error message"
                logger.error(f"Server error (500) uploading novel '{novel.title}': {error_text}")
                try:
                    error_json = response.json()
                    logger.error(f"Server error JSON: {json.dumps(error_json, indent=2)}")
                except:
                    pass
                return None
            else:
                logger.error(
                    f"Unexpected error uploading novel '{novel.title}': "
                    f"Status {response.status_code}, Response: {response.text}"
                )
                return None

        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                logger.error(f"HTTP error uploading novel '{novel.title}': Status {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:2000] if e.response.text else 'No response body'}")
                try:
                    error_json = e.response.json()
                    logger.error(f"Error JSON: {json.dumps(error_json, indent=2)}")
                except:
                    pass
            else:
                logger.error(f"HTTP error uploading novel '{novel.title}': {e}")
            return None
        except requests.exceptions.Timeout as e:
            # Specifically handle timeout errors
            timeout_type = "read" if isinstance(e, requests.exceptions.ReadTimeout) else "connect" if isinstance(e, requests.exceptions.ConnectTimeout) else "general"
            logger.error(
                f"Timeout error ({timeout_type}) uploading novel '{novel.title}' to {ctx.config.api.url}: "
                f"Server did not respond within 300 seconds. "
                f"Error: {e}"
            )
            logger.error(
                f"This usually means your API endpoint is taking too long to process the request. "
                f"Consider: (1) Processing uploads asynchronously, (2) Reducing batch size, "
                f"or (3) Increasing your server's timeout limits."
            )
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error uploading novel '{novel.title}': {e}")
            logger.error(f"Error type: {type(e).__name__}, Error details: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading novel '{novel.title}': {e}", exc_info=True)
            return None

    def _upload_novel_in_batches(self, novel: Novel, user_id: Optional[str], headers: Dict) -> Optional[Dict]:
        """
        Upload novel in batches: first metadata, then chapters in smaller batches.
        This is used when the full payload is too large (413 error).
        """
        logger.info(f"Starting batch upload for novel '{novel.title}'")
        
        # Step 1: Upload novel metadata without chapters
        try:
            data_no_chapters = self._format_novel_data(novel, include_chapters=False)
            logger.info(f"Uploading novel metadata (without chapters) for '{novel.title}'")
            
            response = self.session.post(
                ctx.config.api.url,
                headers=headers,
                json=data_no_chapters,
                timeout=300,
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to upload novel metadata: Status {response.status_code}, Response: {response.text[:500]}")
                return None
            
            result = response.json()
            novel_id_api = result.get('novelId')
            logger.info(f"Novel metadata uploaded successfully. Novel ID: {novel_id_api}")
            
        except Exception as e:
            logger.error(f"Error uploading novel metadata: {e}", exc_info=True)
            return None
        
        # Step 2: Upload chapters in batches (only new chapters that haven't been uploaded)
        all_chapters = ctx.chapters.list(novel_id=novel.id, is_crawled=True)
        all_chapters.sort(key=lambda c: c.serial)
        
        # Filter out already-uploaded chapters
        chapters = [
            c for c in all_chapters
            if not (hasattr(c, 'extra') and isinstance(c.extra, dict) and c.extra.get('api_uploaded') is True)
        ]
        
        if not chapters:
            already_uploaded = len(all_chapters) - len(chapters)
            if already_uploaded > 0:
                logger.info(f"All {already_uploaded} chapters for novel '{novel.title}' have already been uploaded, skipping")
            else:
                logger.warning(f"No chapters to upload for novel {novel.title}")
            return result
        
        already_uploaded = len(all_chapters) - len(chapters)
        if already_uploaded > 0:
            logger.info(f"Skipping {already_uploaded} already-uploaded chapters, uploading {len(chapters)} new chapters for novel '{novel.title}'")
        
        # Determine batch size (50 chapters per batch, or smaller if needed)
        # Vercel has a 4.5MB limit, so we'll use 50 chapters per batch as a safe default
        batch_size = 50
        batches, chapter_objects_list = self._format_chapters_batch(chapters, batch_size=batch_size)
        
        logger.info(f"Uploading {len(chapters)} chapters in {len(batches)} batches (batch size: {batch_size})")
        
        total_uploaded = 0
        
        # Prepare novel metadata for batch payloads (API expects novel object structure)
        status = self._determine_status(novel)
        cover_url = self._get_cover_url(novel)
        # Capitalize genre names for batch uploads too
        raw_genres_batch = novel.tags or []
        capitalized_genres_batch = [genre.title() if genre else genre for genre in raw_genres_batch] if raw_genres_batch else []
        
        novel_metadata = {
            "novel": {
                "title": novel.title,
                "author": novel.authors or "Unknown",
                "description": novel.synopsis or "",
                "coverImageUrl": cover_url,
                "status": status,
                "originalLanguage": self._get_language_name(novel.language),
                "sourceUrl": novel.url,
            },
            "genres": capitalized_genres_batch,
        }
        
        for batch_num, (chapter_batch, chapter_objs) in enumerate(zip(batches, chapter_objects_list), 1):
            try:
                logger.info(f"Uploading batch {batch_num}/{len(batches)} ({len(chapter_batch)} chapters)...")
                
                # Include full novel metadata structure with each batch as required by API
                # Also include novelId to associate chapters with the already-created novel
                batch_data = {
                    **novel_metadata,  # Include novel object and genres
                    "chapters": chapter_batch,
                    "novelId": novel_id_api,  # Associate with the novel created in step 1
                }
                
                # Upload directly to main endpoint (chapters endpoint doesn't exist or has different structure)
                batch_response = self.session.post(
                    ctx.config.api.url,
                    headers=headers,
                    json=batch_data,
                    timeout=300,
                )
                
                if batch_response.status_code == 200:
                    batch_result = batch_response.json()
                    imported = batch_result.get('chaptersImported', len(chapter_batch))
                    total_uploaded += imported
                    logger.info(f"Batch {batch_num}/{len(batches)} uploaded successfully: {imported} chapters imported")
                    
                    # Mark chapters as uploaded in the database
                    self._mark_chapters_uploaded_objects(chapter_objs)
                else:
                    # Check if we got a Vercel Security Checkpoint page
                    batch_response_text = batch_response.text if batch_response.text else ""
                    if "Vercel Security Checkpoint" in batch_response_text or "data-astro-cid" in batch_response_text:
                        logger.error(
                            f"Vercel Security Checkpoint blocked batch {batch_num}/{len(batches)} upload. "
                            f"Status: {batch_response.status_code}"
                        )
                        logger.error(
                            f"To fix this, you may need to whitelist your crawler's IP or disable "
                            f"Vercel Security Checkpoint for your API route."
                        )
                    else:
                        logger.error(f"Failed to upload batch {batch_num}/{len(batches)}: Status {batch_response.status_code}, Response: {batch_response.text[:500]}")
                    # Continue with next batch even if one fails
                    
            except requests.exceptions.Timeout as e:
                timeout_type = "read" if isinstance(e, requests.exceptions.ReadTimeout) else "connect" if isinstance(e, requests.exceptions.ConnectTimeout) else "general"
                logger.error(
                    f"Timeout error ({timeout_type}) uploading batch {batch_num}/{len(batches)} for novel '{novel.title}': "
                    f"Server did not respond within 300 seconds. Error: {e}"
                )
                logger.error(
                    f"Batch upload failed due to timeout. This usually means your API endpoint is taking too long. "
                    f"Consider processing uploads asynchronously or reducing batch size."
                )
                # Continue with next batch
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error uploading batch {batch_num}/{len(batches)}: {e}")
                logger.error(f"Error type: {type(e).__name__}, Error details: {str(e)}")
                # Continue with next batch
            except Exception as e:
                logger.error(f"Error uploading batch {batch_num}/{len(batches)}: {e}", exc_info=True)
                # Continue with next batch
        
        logger.info(f"Batch upload completed for novel '{novel.title}': {total_uploaded}/{len(chapters)} chapters uploaded")
        
        # Return result with updated chapter count and status
        if result:
            result['chaptersImported'] = total_uploaded
            novel_status = self._determine_status(novel)
            result['status'] = novel_status
        return result

    def upload_chapters_only(self, novel_id: str, chapter_ids: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Upload only chapters for an existing novel.

        Args:
            novel_id: ID of the novel
            chapter_ids: Optional list of chapter IDs to upload. If None, uploads all available chapters.

        Returns:
            Response data if successful, None otherwise
        """
        if not ctx.config.api.enabled:
            logger.debug("API upload is disabled")
            return None

        if not ctx.config.api.token:
            logger.warning("API token is not configured, skipping upload")
            return None

        # Get novel from database
        novel = ctx.novels.get(novel_id)
        if not novel:
            logger.error(f"Novel {novel_id} not found")
            return None

        # Get chapters
        if chapter_ids:
            chapters = ctx.chapters.get_many(chapter_ids)
        else:
            chapters = ctx.chapters.list(novel_id=novel_id, is_crawled=True)

        chapters.sort(key=lambda c: c.serial)

        if not chapters:
            logger.warning(f"No chapters to upload for novel {novel.title}")
            return None

        logger.info(f"Uploading {len(chapters)} chapters for novel: {novel.title}")

        # Format chapters
        formatted_chapters = []
        for chapter in chapters:
            if not chapter.is_available:
                logger.warning(f"Chapter {chapter.serial} ({chapter.title}) is not available, skipping")
                continue

            try:
                content = ctx.files.load_text(chapter.content_file)
            except Exception as e:
                logger.error(f"Failed to load chapter {chapter.serial}: {e}")
                continue

            chapter_data = {
                "title": chapter.title,
                "content": content,
                # API expects snake_case `chapter_number`
                "chapter_number": chapter.serial,
            }

            # Add publishedAt if available
            if hasattr(chapter, "created_at") and chapter.created_at:
                # Convert timestamp (milliseconds) to ISO 8601 format
                if isinstance(chapter.created_at, (int, float)):
                    # Timestamp is in milliseconds, convert to seconds
                    timestamp_ms = chapter.created_at
                    timestamp_s = timestamp_ms / 1000.0 if timestamp_ms > 1e10 else timestamp_ms
                    dt = datetime.fromtimestamp(timestamp_s)
                else:
                    dt = chapter.created_at
                chapter_data["publishedAt"] = dt.isoformat() + "Z"

            formatted_chapters.append(chapter_data)

        if not formatted_chapters:
            logger.warning(f"No valid chapters to upload for novel {novel.title}")
            return None

        # Prepare request with browser-like headers to avoid Vercel Security Checkpoint
        base_url = ctx.config.api.url.split("/api")[0] if "/api" in ctx.config.api.url else ctx.config.api.url.rsplit("/", 1)[0]
        headers = {
            "Authorization": f"Bearer {ctx.config.api.token}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": base_url,
            "Referer": base_url + "/",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

        data = {
            "chapters": formatted_chapters,
        }

        # Use chapters-only endpoint if available
        chapters_url = ctx.config.api.url.replace("/novels", "/chapters")
        if chapters_url == ctx.config.api.url:
            # Fallback: use the same endpoint
            chapters_url = ctx.config.api.url

        # Make request
        try:
            response = self.session.post(
                chapters_url,
                headers=headers,
                json=data,
                timeout=300,
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"Successfully uploaded {len(formatted_chapters)} chapters for novel '{novel.title}': "
                    f"Chapters imported: {result.get('chaptersImported', len(formatted_chapters))}"
                )
                return result
            elif response.status_code == 400:
                error_data = response.json()
                logger.error(f"Validation error uploading chapters: {error_data.get('error', 'Unknown error')}")
                return None
            elif response.status_code == 401:
                logger.error("Authentication failed: Invalid API token")
                return None
            elif response.status_code == 500:
                logger.error(f"Server error uploading chapters: {response.text}")
                return None
            else:
                logger.error(
                    f"Unexpected error uploading chapters: "
                    f"Status {response.status_code}, Response: {response.text}"
                )
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error uploading chapters: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading chapters: {e}", exc_info=True)
            return None
